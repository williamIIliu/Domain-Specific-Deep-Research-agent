import json
import os
import warnings
from typing import List, Dict
import torch.nn.functional as F
import functools
from tqdm import tqdm
from multiprocessing import Pool
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
import argparse
import datasets

# ------------------------- 数据读取与映射 -------------------------
def load_docs(corpus, doc_idxs):
    results = [corpus[idx] for idx in doc_idxs]

    return results

def load_corpus(corpus_path: str) -> Dict[str, Dict]:
    """返回字典结构：{doc_id: 文档信息}，方便按id快速查找"""
    corpus = {}  # 键：文档id，值：文档完整信息
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            data = json.loads(line)
            doc_id = data.get("id", "")

            contents = data.get("contents", "")
            if isinstance(contents, dict):
                contents_str = ", ".join([f'"{k}":"{v}"' for k, v in contents.items()])
            else:
                contents_str = str(contents)

            corpus[doc_id] =  contents_str
    return corpus

# ------------------------- Retrieval 基类 -------------------------
class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.top_k
        self.corpus = load_corpus(config.jsonl_path)
        # self.cache_save_path = os.path.join(config.save_dir, 'retrieval_cache.json')

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.
        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)
        """
        pass

    def _batch_search(self, query_list, num, return_score):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

# ------------------------- BM25 Retrieval -------------------------
class BM25Retriever(BaseRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(config.index_bm25_path)
        self.contain_doc = self._check_contain_doc()
        self.max_process_num = 8

    def _check_contain_doc(self):
        r"""Check if the index contains document content
        """
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score=False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk

        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []

        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            results = [{'title': content.split("\n")[0].strip("\""),
                        'text': "\n".join(content.split("\n")[1:]),
                        'contents': content} for content in all_contents]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list, num: int = None, return_score=False):
        # TODO: modify batch method
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)

        if return_score:
            return results, scores
        else:
            return results


# ------------------------- FAISS Retrieval -------------------------
def last_token_pool(last_hidden_states, attention_mask ) :
    """Qwen3专用：提取最后有效token的隐藏状态"""
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
    """Qwen3专用：计算embedding（包含池化和归一化）"""
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    with torch.no_grad():  # 推理模式，禁用梯度
        outputs = model(** batch_dict)
    # 池化+归一化
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Qwen3专用：构造带指令的查询文本"""
    return f'指令: {task_description}\n查询: {query}'


class DenseRetriever(BaseRetriever):
    r"""基于Qwen3 Embedding和Faiss的稠密检索器"""

    def __init__(self, config):
        super().__init__(config)
        # 加载Qwen3模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.embedding_model_path,
            padding_side="left",  # Qwen3专用：左padding
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            config.embedding_model_path,
            torch_dtype=torch.bfloat16  # 使用bf16推理
        )
        self.model.eval()  # 推理模式
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 加载Faiss索引
        self.index = faiss.read_index(args.index_faiss_path)
        self.id_map_path = os.path.join(os.path.dirname(config.index_faiss_path), "id_map.json")
        if not os.path.exists(self.id_map_path):
            raise FileNotFoundError(f"未找到 id_map.json，路径：{self.id_map_path}")
        with open(self.id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)  # self.id_map: [doc_id0, doc_id1, ...]，索引=Faiss返回的idx

        # 检索参数
        self.batch_size = config.retrieval_batch_size
        self.task_description = config.task_desc  # Qwen3指令描述

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Qwen3专用：批量编码查询为向量"""
        # 1. 构造带指令的查询文本
        instructed_queries = [
            get_detailed_instruct(self.task_description, q)
            for q in queries
        ]
        # 2. Tokenize
        batch_dict = self.tokenizer(
            instructed_queries,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.query_max_length
        )
        # 3. 计算embedding（bf16推理）
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 启用bf16自动混合精度
            embeddings = compute_embeddings(self.model, batch_dict, self.device)
        return embeddings.cpu().numpy()  # 转为numpy数组供Faiss使用

    def _search(self, query: str, num: int = None, return_score: bool = False) -> List[Dict] | tuple:
        """单查询检索（适配Qwen3）"""
        num = num or self.topk

        query_emb = self._encode_queries([query])

        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        # print(scores)
        # print(idxs)

        # # 映射文档并补充分数
        doc_ids = []
        for idx in idxs:
            if idx == -1:
                warnings.warn("Faiss未检索到匹配结果，跳过")
                continue
            if 0 <= idx < len(self.id_map):
                doc_ids.append(self.id_map[idx])
            else:
                warnings.warn(f"Faiss索引位置 {idx} 超出 id_map 范围，跳过")

        # 拿到纯文本结果（字符串列表）
        results = load_docs(self.corpus, doc_ids)

        if len(results) < num:
            warnings.warn(f"检索结果不足{num}条，实际返回{len(results)}条")
        return (results, scores) if return_score else results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False) -> List[List[
        Dict]] | tuple:
        """批量查询检索（适配Qwen3）"""
        num = num or self.topk
        if not query_list:
            return ([], []) if return_score else []

        all_results = []
        all_scores = []
        for start in tqdm(range(0, len(query_list), self.batch_size), desc="Qwen3 Faiss批量检索"):
            batch_queries = query_list[start:start + self.batch_size]
            batch_embs = self._encode_queries(batch_queries)
            batch_scores, batch_idxs = self.index.search(batch_embs, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            for idxs, scores in zip(batch_idxs, batch_scores):
                doc_ids = []
                for idx in idxs:
                    if idx == -1:
                        continue
                    if 0 <= idx < len(self.id_map):
                        doc_ids.append(self.id_map[idx])
                # 拿到纯文本结果
                batch_docs = load_docs(self.corpus, doc_ids)
                # 删掉给字符串加score的代码
                all_results.append(batch_docs)
                all_scores.append(scores)
        return (all_results, all_scores) if return_score else all_results




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieval")
    # Path
    parser.add_argument('--index_bm25_path', type=str, default="./datasets/database/bm25")
    parser.add_argument('--index_faiss_path', type=str, default="./datasets/database/faiss_qwen/faiss_index.bin" )
    parser.add_argument('--jsonl_path', type=str,default="./datasets/OmniEval-Corpus/all_data_clean.jsonl")
    parser.add_argument('--log_path', type=str, default="./logs/retrieval")
    parser.add_argument('--embedding_model_path', type=str, default="./pretrain_models/embedding/Qwen3-Embedding-0.6B")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--query_max_length', type=str, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--task_desc', type=str, default="根据给定的搜索查询，检索最相关的段落来回答问题")

    args = parser.parse_args()

    # 读取文本数据
    datasets = load_corpus(args.jsonl_path)
    print(len(datasets))
    print(datasets)

    # BM25 测试
    # bm25_retriever = BM25Retriever(args)
    # query = "在2021年6月15日，基金代码为009730的单位净值、复权单位净值及累计单位净值分别为多少？？"
    # results, scores = bm25_retriever.search(query, return_score=True)
    # print(f"查询：{query}")
    # for i, (res, score) in enumerate(zip(results, scores), 1):
    #     content = res.get('contents', res)[:100]  # 适配两种返回格式
    #     print(f"Top{i} - 分数：{score:.4f}，内容：{content}...")
    # print("=" * 50)

    # FAISS 测试
    dense_retriever = DenseRetriever(args)
    query = "在2021年6月15日，基金代码为009730的单位净值、复权单位净值及累计单位净值分别为多少？"
    results, scores = dense_retriever.search(query, return_score=True)
    print(f"单查询：{query}")
    for i in range(len(results)):
        # 通过索引 i 关联：results[i]是内容，doc_ids[i]是ID，scores[i]是分数
        doc_content = results[i]
        score = scores[i]
        print(f"Top{i + 1} - 分数：{score:.4f}，内容：{doc_content[:100]}...")
    print("=" * 50)

    queries = ["贝因美在面临退市危机时采取了哪些财务措施来改善业绩？",
               "厦门市分局在参加总局自贸区工作视频会议后，郑卫国副局长对下一阶段自贸区工作提出了哪些具体任务和要求？",
               "为什么我国选择粤港澳大湾区作为跨境贸易投资便利化政策的试点地区，这些政策如何体现对香港及该区域的重视？"]
    batch_results, batch_scores = dense_retriever.batch_search(queries, return_score=True)
    print("FAISS批量查询测试：")
    for q_idx, (query, res_list, score_list) in enumerate(zip(queries, batch_results, batch_scores)):
        print(f"\n查询{q_idx + 1}：{query}")
        for r_idx, (doc_content, score) in enumerate(zip(res_list, score_list[:len(res_list)])):
            # 纯文本结果直接打印，无需访问['contents']
            print(f"  Top{r_idx + 1} - 分数：{score:.4f}，内容：{doc_content[:80]}...")

