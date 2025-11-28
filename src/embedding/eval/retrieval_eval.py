"""
检索质量评估脚本
支持 dense, bm25, hybrid 三种检索方式
评估指标: Top3@accuracy, Top5@accuracy, MRR
"""

import json
import os
import argparse
import requests
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any


def load_test_data(test_file: str) -> List[Dict]:
    """加载测试数据集"""
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def retrieve(query: str, topk: int, retrieval_type: str, hybrid_alpha: float, server_url: str) -> List[Dict]:
    """调用检索服务"""
    data = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
        "retrieval_type": retrieval_type,
        "hybrid_alpha": hybrid_alpha
    }
    
    try:
        response = requests.post(
            f"{server_url}/retrieve",
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        result = response.json()
        # 返回第一个query的结果
        # 服务返回格式: {"code": 200, "message": "success", "data": [[{...}, {...}], ...]}
        # data 是二维数组: data[query_idx][result_idx]
        if "data" in result and len(result["data"]) > 0:
            first_query_results = result["data"][0]
            # 如果是二维数组，取第一个query的结果列表
            if isinstance(first_query_results, list):
                return first_query_results
            # 如果直接是结果字典，包装成列表
            return [first_query_results]
        return []
    except Exception as e:
        print(f"检索失败: {e}")
        return []


def evaluate(test_data: List[Dict], retrieval_type: str, hybrid_alpha: float, 
             server_url: str, topk: int = 10) -> Dict[str, Any]:
    """
    评估检索质量
    
    Args:
        test_data: 测试数据列表，每条包含 query 和 id (参考答案)
        retrieval_type: 检索类型 (dense, bm25, hybrid)
        hybrid_alpha: 混合检索权重
        server_url: 检索服务地址
        topk: 检索返回的最大数量
    
    Returns:
        评估指标字典
    """
    total = len(test_data)
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks = []
    
    print(f"\n开始评估 {retrieval_type} 检索方式...")
    print(f"测试样本数: {total}")
    
    # 调试：打印第一个样本的检索结果
    if test_data:
        first_item = test_data[0]
        first_query = first_item.get("query", "")
        first_gt_id = first_item.get("id", "")
        first_results = retrieve(first_query, topk, retrieval_type, hybrid_alpha, server_url)
        print(f"\n[调试] 第一个样本:")
        print(f"  Query: {first_query[:50]}...")
        print(f"  Ground Truth ID: {first_gt_id}")
        print(f"  检索返回数量: {len(first_results)}")
        if first_results:
            print(f"  第一个结果: {first_results[0]}")
    
    for item in tqdm(test_data, desc="评估进度"):
        query = item.get("query", "")
        ground_truth_id = item.get("id", "")
        
        if not query or not ground_truth_id:
            continue
        
        # 获取检索结果
        results = retrieve(query, topk, retrieval_type, hybrid_alpha, server_url)
        
        # 提取返回的文档ID列表
        # 返回格式: [{"id": "xxx", "contents": "...", "score": 0.9}, ...]
        retrieved_ids = []
        for r in results:
            if isinstance(r, dict):
                doc_id = r.get("id", "")
                retrieved_ids.append(doc_id)
            elif isinstance(r, str):
                retrieved_ids.append(r)
        
        # 计算指标
        # Top3 准确率
        if ground_truth_id in retrieved_ids[:3]:
            top3_hits += 1
        
        # Top5 准确率
        if ground_truth_id in retrieved_ids[:5]:
            top5_hits += 1
        
        # MRR (Mean Reciprocal Rank)
        try:
            rank = retrieved_ids.index(ground_truth_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    
    # 计算最终指标
    metrics = {
        "total_queries": total,
        "Top3@accuracy": top3_hits / total if total > 0 else 0,
        "Top5@accuracy": top5_hits / total if total > 0 else 0,
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        "top3_hits": top3_hits,
        "top5_hits": top5_hits
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """打印评估指标"""
    print(f"\n检索质量评估指标:")
    print(f"总查询数: {metrics['total_queries']}")
    print(f"Top3准确率: {metrics['Top3@accuracy']:.4f} ({metrics['Top3@accuracy'] * 100:.2f}%)")
    print(f"Top5准确率: {metrics['Top5@accuracy']:.4f} ({metrics['Top5@accuracy'] * 100:.2f}%)")
    print(f"平均倒数排名(MRR): {metrics['MRR']:.4f}")


def save_results(results: Dict[str, Any], log_file: str):
    """
    保存评估结果到日志文件，累加而非覆盖
    
    Args:
        results: 评估结果
        log_file: 日志文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 读取现有日志
    existing_logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                existing_logs = json.load(f)
                if not isinstance(existing_logs, list):
                    existing_logs = [existing_logs]
        except (json.JSONDecodeError, FileNotFoundError):
            existing_logs = []
    
    # 添加新结果
    existing_logs.append(results)
    
    # 写回文件
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(existing_logs, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="检索质量评估脚本")
    
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="datasets/embedding/retrieval_test_QA.jsonl",
        help="测试数据集路径"
    )
    parser.add_argument(
        "--retrieval_type", 
        type=str, 
        default="dense",
        choices=["dense", "bm25", "hybrid"],
        help="检索方式: dense, bm25, hybrid"
    )
    parser.add_argument(
        "--hybrid_alpha", 
        type=float, 
        default=0.5,
        help="混合检索权重 (0-1), 越大越偏向dense"
    )
    parser.add_argument(
        "--server_url", 
        type=str, 
        default="http://localhost:8080",
        help="检索服务地址"
    )
    parser.add_argument(
        "--topk", 
        type=int, 
        default=10,
        help="检索返回的最大数量"
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        default="logs/embedding/eval.json",
        help="评估结果日志文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_file}")
    test_data = load_test_data(args.test_file)
    print(f"加载完成，共 {len(test_data)} 条测试样本")
    
    # 执行评估
    metrics = evaluate(
        test_data=test_data,
        retrieval_type=args.retrieval_type,
        hybrid_alpha=args.hybrid_alpha,
        server_url=args.server_url,
        topk=args.topk
    )
    
    # 打印结果
    print_metrics(metrics)
    
    # 构建完整结果记录
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "test_file": args.test_file,
            "retrieval_type": args.retrieval_type,
            "hybrid_alpha": args.hybrid_alpha,
            "server_url": args.server_url,
            "topk": args.topk
        },
        "metrics": metrics
    }
    
    # 保存结果
    save_results(results, args.log_file)


if __name__ == "__main__":
    main()
