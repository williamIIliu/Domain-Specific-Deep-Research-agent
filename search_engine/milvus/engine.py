import os
import json
from typing import Any, Dict, List
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
import time
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Data Processor
# -------------------------
class DataProcessor:
    def __init__(self, embedding_model=None, batch_size=64):
        self.embedding_model = embedding_model
        self.embedding_batch_size = batch_size

    def data_map(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """把 JSONL 里的 record 转换成符合 Milvus schema 的格式"""
        rid = record.get("id", "")
        text_chunk = record.get("contents", "")
        metadata = record.get("metadata", {})

        # 时间
        last_modified = metadata.get("last_modified_date")
        publish_time = int(last_modified.replace("-", "")) if last_modified else 0

        # 来源
        source = metadata.get("file_name") or metadata.get("source_file") or ""

        return {
            "id": rid,
            "text_chunk": text_chunk[:1024],
            "publish_time": publish_time,
            "source": source,
            "metadata": {"score": 0.0},  # 强化学习用
        }

    def get_embedding(self, texts: List[str]):
        """批量获取 embeddings"""
        return self.embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False
        ).tolist()

    def load_and_prepare(self, jsonl_file: str, limit: int = 10000) -> List[Dict[str, Any]]:
        """分布式读取 JSONL + 生成 embedding"""
        # Step 1. 初始化分布式
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        # Step 2. 每个进程读一部分数据
        records, texts = [], []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = [line for i, line in enumerate(f) if i < limit]

        # 切分给不同 rank
        chunk = lines[rank::world_size]

        for line in tqdm(chunk, desc=f"Rank {rank} processing", disable=(rank != 0)):
            try:
                record = json.loads(line)
                mapped = self.data_map(record)
                records.append(mapped)
                texts.append(mapped["text_chunk"])
            except Exception as e:
                if rank == 0:
                    tqdm.write(f"❌ JSON 解码失败: {e}")
                continue

        # Step 3. 批量计算 embedding
        if texts:
            embeddings = self.get_embedding(texts)
            for rec, emb in zip(records, embeddings):
                rec["embedding"] = emb
            torch.cuda.empty_cache()  # 清理显存

        return records

# ================= Embedding Utils =================
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    提取最后有效 token
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

def compute_embeddings(model, batch_dict, device):
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

# -------------------------
# 使用示例
# -------------------------
if __name__ == "__main__":
    # -------------------------
    # 处理jsonl数据，生成embedding数据，获得schema格式数据
    # -------------------------

    # model path
    # qwen3_path = "../../pretrain_weights/embedding/qwen3-0_6b"
    # qwen3_finetune_path = "../../pretrain_weights/embedding/qwen3-0_6b_finetune"
    # bgem3_path = "../../pretrain_weights/embedding/bge-m3"
    # bgem3_finetune_path = "../../pretrain_weights/embedding/bge-m3_finetune"
    # gte_path = "../../pretrain_weights/embedding/gte-large-zh"
    # gte_finetune_path = "../../pretrain_weights/embedding/gte-large-zh_finetune"
    #
    # model = SentenceTransformer(qwen3_path, device="cuda")

    # data path
    data_path = "../../datasets/OmniEval-Corpus/all_data_clean_embedding.jsonl" #"/lvdata/lvzqliu/LLM/rag/datasets/OmniEval-Corpus/all_data.jsonl"
    
    # 开始处理数据
    # processor = DataProcessor(embedding_model=model, batch_size=4)  # 减小 batch_size 防 OOM
    # dataset = processor.load_and_prepare(data_path, limit=4)
    #
    # print(f"✅ 一共处理 {len(dataset)} 条数据")
    # print("示例：", dataset[0])

    # -------------------------
    # 创建milvus collection，插入数据，建立索引
    # -------------------------
    # 1. 连接 Milvus
    db_name = "Finance_Corpus"
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus",
        db_name=db_name
    )

    # 删除重建数据库
    collections = client.list_collections()
    for collection_name in collections:
        client.drop_collection(collection_name)
    db_list = client.list_databases()
    for db_name in db_list:
        if db_name != "default":
            client.drop_database(db_name)

    # 重新创建数据库
    db_list = client.list_databases()
    if "Finance_Corpus" not in db_list:
        client.create_database("Finance_Corpus")

    client = MilvusClient(
        uri="http://localhost:19530",
        user="root",
        password="Milvus",
        db_name=db_name
    )

    # 2.定义schema格式
    analyzer_params = {
        "type": "chinese"  # 指定分词器类型为中文
    }
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
        FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=1024, enable_analyzer=True, analyzer_params=analyzer_params, enable_match=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="publish_time", dtype=DataType.INT64),
        FieldSchema(name="metadata", dtype=DataType.JSON, enable_dynamic=True)  # 动态字段存 score
    ]
    schema = CollectionSchema(fields, description="Finance RAG helper collection")


    # 3 定义检索方式 index_params
    # 3.1 创建Embedding索引, 需要和schema中embedding的name一致,
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 1024},  # 分成多少簇, 越多划分越细，检索速度更慢； 参数从32-4096， 我们设置128
    )

    ## 3.2 创建BM25索引
    # index_params.add_index(
    #     field_name="sparse_bm25",
    #     index_name="sparse_bm25_index",
    #     index_type="SPARSE_WAND",
    #     metric_type="BM25"
    # )

    # 4 创建collection 与分区
    collection_name = "Finance_RAG_helper"
    collections = client.list_collections()
    print("当前已有的collections", collections)
    if collection_name not in collections:
        collection = client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )

    # 4.2 创建分区
    partition_name = "base"
    partitions = client.list_partitions(collection_name)
    if partition_name not in partitions:
        client.create_partition(collection_name, partition_name)


    # 5. 插入数据，分区partition
    # for rec in dataset:
    #     print(len(rec["text_chunk"]))
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            dataset.append(record)

    batch_size = 500
    n = len(dataset)

    total_time = 0
    for i in range(0, n, batch_size):
        batch = dataset[i: i + batch_size]
        start = time.time()
        client.insert(
            collection_name=collection_name,
            data=batch,
            partition_name=partition_name
        )
        batch_time = time.time() - start
        total_time += batch_time
        print(f"已插入 {min(i + batch_size, n)} / {n} 条，用时 {batch_time:.2f} 秒")

    print(f"总耗时 {total_time:.2f} 秒，平均 {(total_time / n) * 1000:.2f} ms/条")

    # 5. 检索数据库
    # 加载数据库
    # query_text = "请对比分析2019年9月和10月中国房地产市场的表现有何不同，以及政策效果的持续性如何"
    # client.load_collection(collection_name=collection_name)
    #
    # qwen3_path = "../../pretrain_weights/embedding/qwen3-0_6b"
    # tokenizer = AutoTokenizer.from_pretrained(qwen3_path, padding_side="left", local_files_only=True)
    # model = AutoModel.from_pretrained(qwen3_path)
    # model.eval()
    #
    # # 自动选设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # query_token_id = tokenizer(query_text, return_tensors="pt", truncation=True, padding=True)
    # query_vector = compute_embeddings(model, query_token_id,device)
    # query_list = query_vector[0].detach().cpu().numpy().tolist()
    # print(query_vector)
    # print(query_vector.shape)

    # start_time = time.time()

    # 5.1 语义检索
    # recalls = client.search(
    #     # 在collection集合下base分区，某个field查找
    #     collection_name=collection_name,
    #     partition_names=[partition_name],
    #     anns_field="embedding",
    #     data=[query_list],
    #
    #     # top k
    #     limit=5,
    #
    #     # 过滤检索
    #     # 按照数据来源和时间进行过滤
    #     # filter = 'source in ["news_a", "news_b"] and publish_time >= 20230901',
    #
    #     # 输出结果，
    #     output_fields=["id", "text_chunk", "source", "publish_time"]
    # )


    # 5.2 混合检索
    # search_params_dense = {
    #     "metric_type": "IP",
    #     "params": {"nlist": 1024}
    # }
    # embedding_topk = 5
    # # Create a dense vector search request
    # request_dense = AnnSearchRequest([query_vector[0]], "dense", search_params_dense, limit=embedding_topk)
    #
    #
    # search_params_bm25 = {
    #     "metric_type": "BM25"
    # }
    # BM25_topk = 5
    # request_bm25 = AnnSearchRequest([query_text], "sparse_bm25", search_params_bm25, limit=BM25_topk)
    #
    # # 结合两个检索
    # reqs = [request_dense, request_bm25]
    #
    # # Initialize the RRF ranking algorithm
    # ranker = RRFRanker(100)
    #
    # # Perform the hybrid search
    # hybrid_search_res = client.hybrid_search(
    #     collection_name=collection_name,
    #     reqs=reqs,
    #     ranker=ranker,
    #     limit=embedding_topk+BM25_topk,
    #     output_fields=["text"]
    # )




