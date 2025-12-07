"""统计 Agentar-DeepFinance-100K 数据集的相关信息。"""

import argparse
import json
from typing import Dict, List

import numpy as np


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_question(record: Dict) -> str:
    """从记录中提取 question（user message）。"""
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_answer(record: Dict) -> str:
    """从记录中提取 answer（assistant message）。"""
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def compute_stats(lengths: List[int], thresholds: List[int] = None) -> Dict:
    """计算长度统计信息。"""
    if not lengths:
        return {}
    
    arr = np.array(lengths)
    stats = {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    
    if thresholds:
        for t in thresholds:
            stats[f"exceed_{t}"] = int(np.sum(arr > t))
            stats[f"exceed_{t}_ratio"] = float(np.mean(arr > t))
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="统计数据集信息")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/Agentar-DeepFinance-100K/Agentar_DeepFinance_sft.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="长度阈值列表",
    )
    args = parser.parse_args()

    print(f"加载数据: {args.input}")
    data = load_jsonl(args.input)
    print(f"总数据量: {len(data)}")
    print("=" * 60)

    # 提取 question 和 answer
    questions = [extract_question(r) for r in data]
    answers = [extract_answer(r) for r in data]

    question_lengths = [len(q) for q in questions]
    answer_lengths = [len(a) for a in answers]

    # Question 统计
    print("\n【Question 统计】")
    q_stats = compute_stats(question_lengths, args.thresholds)
    print(f"  数量: {q_stats['count']}")
    print(f"  平均长度: {q_stats['mean']:.2f}")
    print(f"  标准差: {q_stats['std']:.2f}")
    print(f"  最小长度: {q_stats['min']}")
    print(f"  最大长度: {q_stats['max']}")
    print(f"  中位数: {q_stats['median']:.2f}")
    print(f"  P25: {q_stats['p25']:.2f}")
    print(f"  P75: {q_stats['p75']:.2f}")
    print(f"  P90: {q_stats['p90']:.2f}")
    print(f"  P95: {q_stats['p95']:.2f}")
    print(f"  P99: {q_stats['p99']:.2f}")
    print("\n  超出阈值统计:")
    for t in args.thresholds:
        cnt = q_stats.get(f"exceed_{t}", 0)
        ratio = q_stats.get(f"exceed_{t}_ratio", 0)
        print(f"    > {t}: {cnt} ({ratio*100:.2f}%)")

    # Answer 统计
    print("\n【Answer 统计】")
    a_stats = compute_stats(answer_lengths, args.thresholds)
    print(f"  数量: {a_stats['count']}")
    print(f"  平均长度: {a_stats['mean']:.2f}")
    print(f"  标准差: {a_stats['std']:.2f}")
    print(f"  最小长度: {a_stats['min']}")
    print(f"  最大长度: {a_stats['max']}")
    print(f"  中位数: {a_stats['median']:.2f}")
    print(f"  P25: {a_stats['p25']:.2f}")
    print(f"  P75: {a_stats['p75']:.2f}")
    print(f"  P90: {a_stats['p90']:.2f}")
    print(f"  P95: {a_stats['p95']:.2f}")
    print(f"  P99: {a_stats['p99']:.2f}")
    print("\n  超出阈值统计:")
    for t in args.thresholds:
        cnt = a_stats.get(f"exceed_{t}", 0)
        ratio = a_stats.get(f"exceed_{t}_ratio", 0)
        print(f"    > {t}: {cnt} ({ratio*100:.2f}%)")

    # 总长度统计 (question + answer)
    total_lengths = [q + a for q, a in zip(question_lengths, answer_lengths)]
    print("\n【总长度统计 (Question + Answer)】")
    t_stats = compute_stats(total_lengths, args.thresholds)
    print(f"  平均长度: {t_stats['mean']:.2f}")
    print(f"  最小长度: {t_stats['min']}")
    print(f"  最大长度: {t_stats['max']}")
    print(f"  P90: {t_stats['p90']:.2f}")
    print(f"  P95: {t_stats['p95']:.2f}")
    print(f"  P99: {t_stats['p99']:.2f}")
    print("\n  超出阈值统计:")
    for t in args.thresholds:
        cnt = t_stats.get(f"exceed_{t}", 0)
        ratio = t_stats.get(f"exceed_{t}_ratio", 0)
        print(f"    > {t}: {cnt} ({ratio*100:.2f}%)")


if __name__ == "__main__":
    main()
