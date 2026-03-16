import json
import os
import argparse
import re
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
try:
    import jieba
except ImportError:
    jieba = None

_userdict_loaded = False

def tokenize_query(text: str, ner_dict_path: str | None = None, use_jieba: bool = False) -> str:
    """Tokenize query consistent with build_index.py
    1. Noise reduction via regex (same as build_index.py)
    2. Jieba segmentation (no boosting)
    """
    global _userdict_loaded
    
    # 1. 符号降噪：去除 JSON 特殊符号，压缩空格
    text = re.sub(r'["\'{}:,\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    if not jieba:
        return text
        
    # Load dictionary if needed
    if ner_dict_path and not _userdict_loaded:
        if os.path.exists(ner_dict_path):
            try:
                jieba.load_userdict(ner_dict_path)
            except Exception:
                pass
        _userdict_loaded = True

    # Tokenize if using jieba (either via flag or dict presence)
    if use_jieba or ner_dict_path:
        seg_list = list(jieba.cut(text))
        return " ".join([tok for tok in seg_list if tok])
        
    return text

def evaluate_pyserini_bm25(querys_path, index_path, ner_dict_path: str | None = None, use_jieba_pretokenize: bool = False):
    print("\n--- Evaluating Pyserini BM25 (Lucene) ---")
    if not os.path.exists(index_path):
        print(f"Error: Index path {index_path} does not exist.")
        return None

    try:
        searcher = LuceneSearcher(index_path)
        # Match analyzer with indexing: if pretokenized (jieba userdict OR jieba-only), do not set zh analyzer
        if not (((ner_dict_path and jieba) or (use_jieba_pretokenize and jieba))):
            searcher.set_language('zh')
    except Exception as e:
        print(f"Error initializing Pyserini searcher: {e}")
        return None

    total = 0
    top1_hit = 0
    top3_hit = 0
    top5_hit = 0

    with open(querys_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Pyserini BM25"):
            data = json.loads(line)
            query = data.get('query')
            true_id = data.get('id')
            if not query or not true_id: continue
            
            total += 1
            tokenized_q = tokenize_query(query, ner_dict_path, use_jieba=use_jieba_pretokenize)
            hits = searcher.search(tokenized_q, k=5)
            hit_ids = [hit.docid for hit in hits]
            
            if len(hit_ids) >= 1 and true_id == hit_ids[0]: top1_hit += 1
            if true_id in hit_ids[:3]: top3_hit += 1
            if true_id in hit_ids[:5]: top5_hit += 1

    results = {"top1": top1_hit/total, "top3": top3_hit/total, "top5": top5_hit/total, "total": total}
    print_metrics(results)
    return results

def print_metrics(res):
    if res:
        print(f"Total Queries: {res['total']}")
        print(f"Top-1 Accuracy: {res['top1']:.4f}")
        print(f"Top-3 Accuracy: {res['top3']:.4f}")
        print(f"Top-5 Accuracy: {res['top5']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval with different tokenization modes")
    parser.add_argument("--queries", default="/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/datasets/embedding/querys.jsonl", help="Path to queries jsonl")
    parser.add_argument("--index", required=False, default="/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/datasets/database/bm25_index_native", help="Path to Lucene index")
    parser.add_argument("--ner-dict", default=None, help="Path to NER user dictionary (for with_ner mode)")
    parser.add_argument("--mode", choices=["auto", "with_ner", "wo_ner", "jieba_only"], default="auto", help="Evaluation mode")
    args = parser.parse_args()

    # Determine mode
    mode = args.mode
    if mode == "auto":
        mode = "with_ner" if args.ner_dict else "wo_ner"

    if mode == "with_ner":
        if not args.ner_dict:
            raise SystemExit("with_ner mode requires --ner-dict path")
        evaluate_pyserini_bm25(args.queries, args.index, ner_dict_path=args.ner_dict, use_jieba_pretokenize=False)
    elif mode == "wo_ner":
        # Match build_index_wo_ner.py: Lucene zh analyzer, no pretokenization
        evaluate_pyserini_bm25(args.queries, args.index, ner_dict_path=None, use_jieba_pretokenize=False)
    elif mode == "jieba_only":
        # Optional: queries pretokenized with jieba only (no dict); analyzer off
        evaluate_pyserini_bm25(args.queries, args.index, ner_dict_path=None, use_jieba_pretokenize=True)
