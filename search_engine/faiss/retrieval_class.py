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

# ------------------------- 数据读取与映射 -------------------------
def load_docs(corpus, doc_ids):
    results = []
    for doc_id in doc_ids:
        if doc_id in corpus:
            results.append({
                "id": doc_id,
                "contents": corpus[doc_id]
            })
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
        self.topk = config.top_k
        self.corpus = load_corpus(config.jsonl_path)
        # self.cache_save_path = os.path.join(config.save_dir, 'retrieval_cache.json')

    def _search(self, query: str, num: int) -> List[Dict[str, str]]:
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

    def _search(self, query: str, num: int = None) -> List[Dict]:
        """执行单查询BM25检索，返回包含id、score、contents的结构化结果"""
        num = num or self.topk
        hits = self.searcher.search(query, num)
        if not hits:
            return []

        results = []
        scores = []

        for hit in hits[:num]:
            try:
                # 解析Lucene文档
                raw_json = self.searcher.doc(hit.docid).raw()
                if raw_json:  # 如果索引中包含完整文档
                    doc_json = json.loads(raw_json)
                    doc_id = doc_json.get("id", str(hit.docid))
                    contents = doc_json.get("contents", "")
                else:
                    # 若索引中不含内容，则从本地corpus查找
                    doc_id = str(hit.docid)
                    contents = self.corpus.get(doc_id, "")
            except Exception:
                # 防止个别文档损坏
                doc_id = str(hit.docid)
                contents = self.corpus.get(doc_id, "")

            # 统一结构化输出
            results.append({
                "id": doc_id,
                "contents": contents,
                "score": hit.score
            })
            scores.append(hit.score)

        if len(results) < num:
            warnings.warn(f"检索结果数量不足 {num} 条，实际返回 {len(results)} 条。")

        return  results

    def _batch_search(self, query_list, num: int = None):
        results = []
        for query in query_list:
            item_result = self._search(query, num, True)
            results.append(item_result)
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
        self.index = faiss.read_index(config.index_faiss_path)
        self.id_map_path = os.path.join(os.path.dirname(config.index_faiss_path), "id_map.json")
        if not os.path.exists(self.id_map_path):
            raise FileNotFoundError(f"未找到 id_map.json，路径：{self.id_map_path}")
        with open(self.id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)  # self.id_map: [doc_id0, doc_id1, ...]，索引=Faiss返回的idx

        # 检索参数
        self.batch_size = config.batch_size
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

    def _search(self, query: str, num: int = None) -> List[Dict] | tuple:
        """单查询检索（适配Qwen3）"""
        num = num or self.topk

        query_emb = self._encode_queries([query])

        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

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
        for r, s in zip(results, scores):
            r["score"] = s  # 给每个文档加上得分

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results

    def _batch_search(self, query_list: List[str], num: int = None) -> List[List[
        Dict]] | tuple:
        """批量查询检索（适配Qwen3）"""
        num = num or self.topk
        if not query_list:
            return []

        all_results = []
        all_scores = []
        for start in tqdm(range(0, len(query_list), self.batch_size), desc="Qwen3 Faiss批量检索"):
            batch_queries = query_list[start:start + self.batch_size]
            batch_embs = self._encode_queries(batch_queries)
            batch_results = self.index.search(batch_embs, k=num)
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
        return all_results


# ------------------------- 混合检索类（BM25 + FAISS） -------------------------
class HybridRetriever(BaseRetriever):
    """混合检索器：融合BM25和稠密检索结果，通过alpha权重重新排序"""

    def __init__(self, config):
        super().__init__(config)
        # 初始化两种检索器
        self.bm25_retriever = BM25Retriever(config)
        self.dense_retriever = DenseRetriever(config)
        # 权重参数：alpha越大，稠密检索权重越高；(1-alpha)为BM25权重
        self.alpha = config.alpha
        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha必须在0到1之间（0=纯BM25，1=纯稠密检索）")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化分数到0-1范围（处理不同检索分数范围差异）"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.5 for _ in scores]  # 分数相同则均为0.5
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _merge_results(self, bm25_docs, bm25_scores, dense_docs, dense_scores, bm25_ids, dense_ids) -> List[Dict]:
        """
        合并两种检索结果（基于排名的 Reciprocal Rank Fusion）：
        - 使用 RRF 公式：score = sum(1 / (k + rank))，其中 k 是平滑常数
        - 排名越靠前，得分越高
        - 最终按融合分数倒序排序
        """
        k = 60  # RRF 平滑常数，常用值为 60
        
        # 构建文档ID到内容的映射（去重）
        doc_id_to_content = {}
        for doc_id, content in zip(bm25_ids, bm25_docs):
            doc_id_to_content[doc_id] = content
        for doc_id, content in zip(dense_ids, dense_docs):
            doc_id_to_content[doc_id] = content

        # 分别计算两个检索器的 RRF 分数
        bm25_rrf_scores = {}
        dense_rrf_scores = {}
        
        # BM25 RRF 分数（排名从1开始）
        for rank, doc_id in enumerate(bm25_ids, start=1):
            bm25_rrf_scores[doc_id] = 1.0 / (k + rank)
        
        # Dense RRF 分数（排名从1开始）
        for rank, doc_id in enumerate(dense_ids, start=1):
            dense_rrf_scores[doc_id] = 1.0 / (k + rank)
        
        # 获取所有文档ID
        all_doc_ids = set(bm25_rrf_scores.keys()) | set(dense_rrf_scores.keys())
        
        # 加权融合：final_score = alpha * dense_rrf + (1 - alpha) * bm25_rrf
        doc_id_to_score = {}
        for doc_id in all_doc_ids:
            bm25_score = bm25_rrf_scores.get(doc_id, 0.0)
            dense_score = dense_rrf_scores.get(doc_id, 0.0)
            doc_id_to_score[doc_id] = self.alpha * dense_score + (1 - self.alpha) * bm25_score

        # 按融合分数倒序排序
        sorted_doc_ids = sorted(doc_id_to_score.items(), key=lambda x: x[1], reverse=True)

        # 将排序后的ID映射回文档内容
        sorted_results = []
        for doc_id, score in sorted_doc_ids[:self.topk]:
            result = {
                "id": doc_id,
                "contents": doc_id_to_content[doc_id],
                "score": score
            }
            sorted_results.append(result)

        return sorted_results

    def _search(self, query: str, num: int = None) -> tuple:
        """单查询混合检索：先分别检索，再合并排序"""
        num = num or self.topk

        # BM25检索（需额外获取原始hits以提取docid）
        bm25_hits = self.bm25_retriever._search(query, num)
        bm25_docs = [hit["contents"] for hit in bm25_hits]
        bm25_scores = [hit["score"] for hit in bm25_hits]
        bm25_ids = [hit["id"] for hit in bm25_hits]

        # 稠密检索（从id_map中获取docid）
        dense_hits = self.dense_retriever._search(query, num=num)
        dense_docs = [hit["contents"] for hit in dense_hits]
        dense_scores = [hit["score"] for hit in dense_hits]
        dense_ids = [hit["id"] for hit in dense_hits]

        # 2. 合并结果并排序
        merged_results = self._merge_results(
            bm25_docs, bm25_scores, dense_docs, dense_scores, bm25_ids, dense_ids
        )

        return merged_results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False) -> tuple:
        """批量查询混合检索：循环处理每个查询"""
        num = num or self.topk
        all_merged_docs = []
        all_merged_scores = []

        for query in tqdm(query_list, desc="混合检索批量处理"):
            merged_docs, merged_scores = self._search(query, num)
            all_merged_docs.append(merged_docs)
            all_merged_scores.append(merged_scores)

        return all_merged_docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieval")
    # Path
    parser.add_argument('--index_bm25_path', type=str, default="../datasets/database/bm25")
    parser.add_argument('--index_faiss_path', type=str, default="../datasets/database/qwen3/faiss_index.bin" )
    parser.add_argument('--jsonl_path', type=str,default="../datasets/Fin_Corpus/clean.jsonl")
    parser.add_argument('--log_path', type=str, default="../logs/retrieval")
    parser.add_argument('--embedding_model_path', type=str, default="../pretrain_weights/embedding/qwen3-0_6b")

    # Basic parameters
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--query_max_length', type=str, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--task_desc', type=str, default="根据给定的搜索查询，检索最相关的段落来回答问题")

    args = parser.parse_args()

    # 读取文本数据
    datasets = load_corpus(args.jsonl_path)
    print(len(datasets))
    # print(datasets)

    # BM25 测试
    bm25_retriever = BM25Retriever(args)
    query = "在2021年6月15日，基金代码为009730的单位净值、复权单位净值及累计单位净值分别为多少？？"
    results = bm25_retriever.search(query)
    print(f"查询：{query}")
    print(f"结果{results}")

    # FAISS 测试
    dense_retriever = DenseRetriever(args)
    query = "在2021年6月15日，基金代码为009730的单位净值、复权单位净值及累计单位净值分别为多少？"
    results = dense_retriever.search(query)
    print(f"查询：{query}")
    print(f"结果{results}")

    # Hybird 测试
    hybrid_retriever = HybridRetriever(args)
    query = "在2021年6月15日，基金代码为009730的单位净值、复权单位净值及累计单位净值分别为多少？"
    results = hybrid_retriever.search(query)
    print(f"查询：{query}")
    print(f"结果{results}")


    #
    # queries = ["贝因美在面临退市危机时采取了哪些财务措施来改善业绩？",
    #            "厦门市分局在参加总局自贸区工作视频会议后，郑卫国副局长对下一阶段自贸区工作提出了哪些具体任务和要求？",
    #            "为什么我国选择粤港澳大湾区作为跨境贸易投资便利化政策的试点地区，这些政策如何体现对香港及该区域的重视？"]
    # batch_results = dense_retriever.batch_search(queries)
    # print("FAISS批量查询测试：")
    # for q_idx, (query, res_list, score_list) in enumerate(zip(queries, batch_results, batch_scores)):
    #     print(f"\n查询{q_idx + 1}：{query}")
    #     for r_idx, (doc_content, score) in enumerate(zip(res_list, score_list[:len(res_list)])):
    #         # 纯文本结果直接打印，无需访问['contents']
    #         print(f"  Top{r_idx + 1} - 分数：{score:.4f}，内容：{doc_content[:80]}...")

    # 混合检索测试（alpha=0.5，平衡两种种检索）
    # hybrid_retriever = HybridRetriever(args)
    #
    # query = "在2021年6月15日，基金代码为009730的单位净值、复权单位净值及累计单位净值分别为多少？"
    # results, scores = hybrid_retriever.search(query, return_score=True)
    # print(f"混合检索单查询（alpha={args.hybrid_alpha}）：{query}")
    # for i, (content, score) in enumerate(zip(results, scores), 1):
    #     print(f"Top{i} - 融合分数：{score:.4f}，内容：{content[:100]}...")
    # print("=" * 50)
    #
    # queries = [
    #     "贝因美在面临退市危机时采取了哪些些财务措施来改善业绩？",
    #     "厦门市分局在参加加总局自贸区工作视频会议后，郑卫国副局长对下一阶段自贸区贸区工作提出出了哪些具体任务和要求？",
    #     "为什么什么我国选择粤港澳大湾区作为跨境贸易投资便利化政策的试点地区，这些政策如何体现对香港及该区域的重视？"
    # ]
    # batch_results, batch_scores = hybrid_retriever.batch_search(queries, return_score=True)
    # print(f"混合检索批量量查询（alpha={args.hybrid_alpha}）：")
    # for q_idx, (query, res_list, score_list) in enumerate(zip(queries, batch_results, batch_scores)):
    #     print(f"\n查询{q_idx + 1}：{query}")
    #     for r_idx, (content, score) in enumerate(zip(res_list, score_list[:len(res_list)])):
    #         print(f"  Top{r_idx + 1} - 融合分数：{score:.4f}，内容：{content[:80]}...")
    #
