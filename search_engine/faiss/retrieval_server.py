import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from retrieval_class import BaseRetriever, DenseRetriever, HybridRetriever, BM25Retriever

# ------------------------- FastAPI 请求模型 -------------------------
class QueryRequest(BaseModel):
    queries: List[str]                 # 批量查询列表
    topk: Optional[int] = None         # 可选，默认使用配置的 top_k
    retrieval_type: Optional[str] = "hybrid"  # bm25 / dense / hybrid
    hybrid_alpha: Optional[float] = 0.5      # 混合检索权重

# ------------------------- 配置类 -------------------------
class Config:
    def __init__(self, **kwargs):
        self.top_k = kwargs.get("top_k", 5)
        self.jsonl_path = kwargs.get("jsonl_path", "")
        self.query_max_length = kwargs.get("query_max_length", 256)
        self.batch_size = kwargs.get("batch_size", 512)
        self.task_desc = kwargs.get("task_desc", "根据给定的搜索查询，检索最相关的段落来回答问题")

        self.index_bm25_path = kwargs.get("index_bm25_path", "")
        self.index_faiss_path = kwargs.get("index_faiss_path", "")
        self.embedding_model_path = kwargs.get("embedding_model_path", "")
        self.alpha = kwargs.get("alpha", 0.5)  # 默认混合权重

# ------------------------- 检索器工厂 -------------------------
def get_retriever(config: Config, retrieval_type: str = "hybrid", hybrid_alpha: Optional[float] = None) -> BaseRetriever:
    if retrieval_type == "bm25":
        return BM25Retriever(config)
    elif retrieval_type == "dense":
        return DenseRetriever(config)
    elif retrieval_type == "hybrid":
        if hybrid_alpha is not None:
            config.alpha = hybrid_alpha
        return HybridRetriever(config)
    else:
        raise ValueError(f"不支持的检索类型：{retrieval_type}，可选 bm25 / dense / hybrid")

# ------------------------- FastAPI 初始化 -------------------------
app = FastAPI(title="多类型检索服务", description="支持 BM25 / Dense / Hybrid 检索")

global_config: Config = None
retriever_cache: Dict[str, BaseRetriever] = {}

# ------------------------- 核心检索接口 -------------------------
@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    global global_config, retriever_cache

    topk = request.topk or global_config.top_k
    retrieval_type = request.retrieval_type.lower()
    hybrid_alpha = request.hybrid_alpha or global_config.alpha

    # 构建缓存 key
    cache_key = f"{retrieval_type}_{hybrid_alpha}" if retrieval_type == "hybrid" else retrieval_type

    # 延迟加载检索器（首次请求时才初始化大模型）
    if cache_key not in retriever_cache:
        retriever_cache[cache_key] = get_retriever(global_config, retrieval_type, hybrid_alpha)

    retriever = retriever_cache[cache_key]

    # 执行批量检索
    results = []
    for query in request.queries:
        result = retriever._search(
            query=query,
            num=topk,
        )
        results.append(result)
    return {"code": 200, "message": "success", "data": results}

# ------------------------- 健康检查 -------------------------
@app.get("/health")
def health_check():
    return {"code": 200, "message": "检索服务运行正常"}

# ------------------------- 启动 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动多类型检索FastAPI服务")

    # 路径参数
    parser.add_argument('--index_bm25_path', type=str, default="../datasets/database/bm25")
    parser.add_argument('--index_faiss_path', type=str, default="../datasets/database/qwen3/faiss_index.bin")
    parser.add_argument('--jsonl_path', type=str, default="../datasets/Fin_Corpus/clean_embedding.jsonl")
    parser.add_argument('--embedding_model_path', type=str, default="../pretrain_weights/embedding/qwen3-0_6b")

    # 服务参数
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=8020)

    # 检索参数
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--query_max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--task_desc', type=str, default="根据给定的搜索查询，检索最相关的段落来回答问题")
    parser.add_argument('--retrieval_method', type=str, default="dense", help="bm25/dense/hybrid")

    args = parser.parse_args()

    # 初始化全局配置
    global_config = Config(
        top_k=args.top_k,
        jsonl_path=args.jsonl_path,
        query_max_length=args.query_max_length,
        batch_size=args.batch_size,
        task_desc=args.task_desc,
        index_bm25_path=args.index_bm25_path,
        index_faiss_path=args.index_faiss_path,
        embedding_model_path=args.embedding_model_path,
        alpha=args.alpha
    )

    print(f"🚀 检索服务启动成功！访问 http://{args.host}:{args.port}/docs 查看API文档")
    uvicorn.run(app, host=args.host, port=args.port)


    # 测试服务
    url = "http://localhost:8020/retrieve"

    data = {
        "queries": ["基金代码009730在2021年6月15日的单位净值是多少？"],
        "topk": 3,
        "retrieval_type": "dense",
        "hybrid_alpha": 0.5
    }

    response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))