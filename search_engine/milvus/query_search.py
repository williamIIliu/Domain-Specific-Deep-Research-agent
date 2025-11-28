import os
import json
from typing import Any, Dict, List
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection
import time
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


# -------------------------
# 创建milvus collection，插入数据，建立索引
# -------------------------
db_name = "Finance_Corpus"
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name=db_name
)

partition_name = "base"
collection_name = "Finance_RAG_helper"
index_name = "embedding"


# 对于不同索引方式的测试
# 1. IVF-FLAT
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 1024}, # 分成多少簇, 越多划分越细，检索速度更慢； 参数从32-4096， 我们设置128
    )
# 2. IVF-PQ
# index_params = MilvusClient.prepare_index_params()
# index_params.add_index(
#     field_name="embedding", # Name of the vector field to be indexed
#     index_type="IVF_PQ", # Type of the index to create
#     metric_type="L2", # Metric type used to measure similarity
#     params={
#         "m": 4, # Number of sub-vectors to split eahc vector into
#     }
# )

# 2. HNSW
# index_params.add_index(
#     field_name="embedding",
#     index_type="HNSW",
#     metric_type="COSINE",
#     params={"M": 16, "efConstruction": 200}, # M 是最大邻居数，越大越慢，通常 8~48，根据数据量和硬件调节，构建索引时的 搜索深度（或者说候选邻居数）
# )
# index_name = "embedding"
# client.release_collection(collection_name=collection_name)
# client.drop_index(collection_name="Finance_RAG_helper", index_name=index_name)
#
# indexes = client.list_indexes(collection_name)
# client.create_index(
#     collection_name="Finance_RAG_helper",
#     index_params=index_params,
#     sync=True
# )


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

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'指令: {task_description}\n查询: {query}'



query_text = "请对比分析2019年9月和10月中国房地产市场的表现有何不同，以及政策效果的持续性如何"
client.load_collection(collection_name=collection_name)
qwen3_path = "../../pretrain_weights/embedding/qwen3-0_6b",

tokenizer = AutoTokenizer.from_pretrained(qwen3_path, padding_side="left", local_files_only=True)
model = AutoModel.from_pretrained(qwen3_path)
model.eval()

# 自动选设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
query_token_id = tokenizer(query_text, return_tensors="pt", truncation=True, padding=True)
query_vector = compute_embeddings(model, query_token_id,device)
query_list = query_vector[0].detach().cpu().numpy().tolist()
print(query_vector)
print(query_vector.shape)

start_time = time.time()
recalls = client.search(
    # 在collection集合下base分区，某个field查找
    collection_name=collection_name,
    partition_names=[partition_name],
    anns_field=index_name,
    data=[query_list],

    # top k
    limit=5,

     #过滤检索
     #按照数据来源和时间进行过滤
     filter = 'source in ["news_a", "news_b"] and publish_time >= 20230901',

     #输出结果，
    output_fields=["id", "text_chunk", "source", "publish_time"]
    )
print(recalls)
print(time.time() - start_time)














# 加载数据库
client.load_collection(collection_name=collection_name)

# 数据
# data_path = "../../datasets/OmniEval-Corpus/qwen_raw.jsonl" #"/lvdata/lvzqliu/LLM/rag/datasets/OmniEval-Corpus/all_data.jsonl"


# 模型
qwen3_path = "../../pretrain_weights/embedding/qwen3-0_6b"
qwen3_finetune_path = "../../pretrain_weights/embedding/qwen3-0_6b_finetune"
bgem3_path = "../../pretrain_weights/embedding/bge-m3"
bgem3_finetune_path = "../../pretrain_weights/embedding/bge-m3_finetune"
gte_path = "../../pretrain_weights/embedding/gte-large-zh"
gte_finetune_path = "../../pretrain_weights/embedding/gte-large-zh_finetune"

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=qwen3_path,
    local_files_only=True,
    # attn_implementation="flash_attention_2",
    dtype=torch.float16
    ).cuda()
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=qwen3_path,
    max_length=2048)
model.eval()
device = "cuda"
# embedding = model.encode(
#     ["你好，世界！"],
#     output_dim=256   # 将 embedding 调整为 256 维
# )

query_text = "请对比分析2019年9月和10月中国房地产市场的表现有何不同，以及政策效果的持续性如何"
task_insturct = "根据给定的搜索查询，检索最相关的段落来回答问题"
query_instruct = get_detailed_instruct(task_insturct, query_text)
print(query_instruct)

query_token_id = tokenizer(query_instruct, return_tensors="pt", truncation=True, padding=True)
query_vector = compute_embeddings(model, query_token_id, device)
query_list = query_vector[0].detach().cpu().numpy().tolist()
print(query_vector)
print(query_vector.shape)

start_time = time.time()
recalls = client.search(
    # 在collection集合下base分区，某个field查找
    collection_name=collection_name,
    partition_names=[partition_name],
    anns_field=index_name,
    data=[query_list],

    # top k
    limit=10,

    # 过滤检索
    # 按照数据来源和时间进行过滤
    # filter = 'source in ["news_a", "news_b"] and publish_time >= 20230901',

    # 输出结果，
    output_fields=["id", "text_chunk", "source", "publish_time"]
)
print("Used time", time.time()-start_time)
print(recalls)