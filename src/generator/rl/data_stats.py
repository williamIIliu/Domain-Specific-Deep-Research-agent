"""统计 RL 数据集 (parquet 格式) 的相关信息。

RL 数据集结构:
- data_source: str
- prompt: List[Dict] (messages 格式)
- ability: str
- reward_model: Dict (包含 ground_truth, style)
- extra_info: Dict (包含 answer, index, question, task_type)
"""

import argparse
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd


def extract_prompt_content(prompt) -> str:
    """从 prompt 中提取用户问题内容。"""
    if isinstance(prompt, (list, np.ndarray)):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return ""


def extract_ground_truth(reward_model) -> str:
    """从 reward_model 中提取 ground_truth。"""
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth", ""))
    return ""


def extract_task_type(extra_info) -> str:
    """从 extra_info 中提取 task_type。"""
    if isinstance(extra_info, dict):
        return str(extra_info.get("task_type", "unknown"))
    return "unknown"


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
    parser = argparse.ArgumentParser(description="统计 RL 数据集信息")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/Agentar-DeepFinance-100K/rl",
        help="RL 数据集目录 (包含 train.parquet 和 test.parquet)",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
        help="长度阈值列表",
    )
    args = parser.parse_args()

    # 加载数据
    train_path = f"{args.input_dir}/train.parquet"
    test_path = f"{args.input_dir}/test.parquet"
    
    print(f"加载训练集: {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"加载测试集: {test_path}")
    test_df = pd.read_parquet(test_path)
    
    print("=" * 60)
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"总数据量: {len(train_df) + len(test_df)}")
    print("=" * 60)

    # 合并数据进行统计
    df = pd.concat([train_df, test_df], ignore_index=True)

    # 提取各字段
    prompts = [extract_prompt_content(p) for p in df["prompt"]]
    ground_truths = [extract_ground_truth(r) for r in df["reward_model"]]
    task_types = [extract_task_type(e) for e in df["extra_info"]]

    prompt_lengths = [len(p) for p in prompts]
    gt_lengths = [len(g) for g in ground_truths]

    # Prompt 统计
    print("\n【Prompt 统计】")
    p_stats = compute_stats(prompt_lengths, args.thresholds)
    print(f"  数量: {p_stats['count']}")
    print(f"  平均长度: {p_stats['mean']:.2f}")
    print(f"  标准差: {p_stats['std']:.2f}")
    print(f"  最小长度: {p_stats['min']}")
    print(f"  最大长度: {p_stats['max']}")
    print(f"  中位数: {p_stats['median']:.2f}")
    print(f"  P25: {p_stats['p25']:.2f}")
    print(f"  P75: {p_stats['p75']:.2f}")
    print(f"  P90: {p_stats['p90']:.2f}")
    print(f"  P95: {p_stats['p95']:.2f}")
    print(f"  P99: {p_stats['p99']:.2f}")
    print("\n  超出阈值统计:")
    for t in args.thresholds:
        cnt = p_stats.get(f"exceed_{t}", 0)
        ratio = p_stats.get(f"exceed_{t}_ratio", 0)
        print(f"    > {t}: {cnt} ({ratio*100:.2f}%)")

    # Ground Truth 统计
    print("\n【Ground Truth 统计】")
    gt_stats = compute_stats(gt_lengths, args.thresholds)
    print(f"  数量: {gt_stats['count']}")
    print(f"  平均长度: {gt_stats['mean']:.2f}")
    print(f"  最小长度: {gt_stats['min']}")
    print(f"  最大长度: {gt_stats['max']}")

    # Task Type 分布
    print("\n【Task Type 分布】")
    task_counter = Counter(task_types)
    for task_type, count in task_counter.most_common():
        ratio = count / len(task_types) * 100
        print(f"  {task_type}: {count} ({ratio:.2f}%)")

    # Ability 分布
    print("\n【Ability 分布】")
    ability_counter = Counter(df["ability"].tolist())
    for ability, count in ability_counter.most_common():
        ratio = count / len(df) * 100
        print(f"  {ability}: {count} ({ratio:.2f}%)")

    # Data Source 分布
    print("\n【Data Source 分布】")
    ds_counter = Counter(df["data_source"].tolist())
    for ds, count in ds_counter.most_common():
        ratio = count / len(df) * 100
        print(f"  {ds}: {count} ({ratio:.2f}%)")


if __name__ == "__main__":
    main()
