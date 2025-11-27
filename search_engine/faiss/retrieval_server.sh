#!/bin/bash
# 检索服务启动脚本（默认使用配置的路径，支持自定义参数）

# 脚本说明：
# 1. 直接运行：使用所有默认路径和参数
# 2. 自定义参数：在脚本后追加参数，例如：./start_retrieval_server.sh --port 8080 --alpha 0.7

 python search_engine/faiss/retrieval_server.py \
   --port 8080 \
   --alpha 0.6 \
   --top_k 5 \
   --index_bm25_path "./datasets/database/bm25" \
   --index_faiss_path "./datasets/database/faiss_qwen/faiss_index.bin" \
   --task_desc "根据给定的搜索查询，检索最相关的段落来回答问题" \
   --jsonl_path "./datasets/OmniEval-Corpus/all_data_clean.jsonl" \
   --embedding_model_path "./pretrain_models/embedding/Qwen3-Embedding-0.6B" \
   --retrieval_method "dense"