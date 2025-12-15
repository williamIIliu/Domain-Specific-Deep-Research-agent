# Dense 检索评估
#python src/embedding/eval/retrieval_eval.py --retrieval_type dense

# BM25 检索评估
#python src/embedding/eval/retrieval_eval.py --retrieval_type bm25

# Hybrid 混合检索评估 (可调整权重)
python src/embedding/eval/retrieval_eval.py --retrieval_type hybrid --hybrid_alpha 0.7
