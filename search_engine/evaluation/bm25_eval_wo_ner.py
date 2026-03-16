import json
import os
import argparse
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

def evaluate_pyserini_bm25(querys_path, index_path):
    print("\n--- Evaluating Pyserini BM25 (Lucene) ---")
    if not os.path.exists(index_path):
        print(f"Error: Index path {index_path} does not exist.")
        return None

    try:
        searcher = LuceneSearcher(index_path)
        # Match build_index_wo_ner.py: use Lucene zh analyzer (no pretokenization)
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
            # Raw query; analyzer handles segmentation
            hits = searcher.search(query, k=5)
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
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval (wo_ner): Lucene zh analyzer, raw queries")
    parser.add_argument("--queries", default="/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/datasets/embedding/querys.jsonl", help="Path to queries jsonl")
    parser.add_argument("--index", default="/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/datasets/database/bm25_index_native", help="Path to Lucene index (wo_ner)")
    args = parser.parse_args()

    evaluate_pyserini_bm25(args.queries, args.index)
