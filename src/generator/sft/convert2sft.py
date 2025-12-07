#!/usr/bin/env python3
"""
Convert Agentar-DeepFinance JSONL to verl SFT parquet format.

verl SFT格式要求:
- question: 用户问题 (字符串)
- answer: 模型回答 (字符串)

Usage:
    python src/generator/sft/convert2sft.py \
        --input_file datasets/Agentar-DeepFinance-100K/Agentar_DeepFinance_sft.jsonl \
        --output_dir datasets/Agentar-DeepFinance-100K \
        --train_ratio 0.95
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def convert_message_to_sft_format(item: dict) -> dict:
    """
    将单条数据转换为verl SFT格式。
    
    verl SFT需要的格式:
    - question: str (用户问题)
    - answer: str (模型回答)
    """
    messages = item.get("messages", [])
    
    # 提取用户问题和助手回答
    question = ""
    answer = ""
    
    for msg in messages:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        
        if role == "HUMAN" or role == "USER":
            question = content
        elif role == "ASSISTANT":
            answer = content
    
    # verl SFT格式: 简单的question/answer对
    return {
        "question": question,
        "answer": answer,
    }


def load_jsonl_file(input_file: str) -> list:
    """
    加载指定的JSONL文件。
    """
    all_data = []
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading {input_path.name}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading {input_path.name}"):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    all_data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    
    return all_data


def main():
    parser = argparse.ArgumentParser(description="Convert DeepFinance JSONL to verl parquet format")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Input JSONL file path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--train_ratio", type=float, default=0.95,
                        help="Ratio of data for training (default: 0.95)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Filter samples where question or answer length exceeds this value")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("Loading data...")
    all_data = load_jsonl_file(args.input_file)
    print(f"Loaded {len(all_data)} samples")

    # 长度过滤
    if args.max_length:
        print(f"Filtering samples with length > {args.max_length}...")
        filtered_data = []
        for item in tqdm(all_data, desc="Filtering"):
            sft_item = convert_message_to_sft_format(item)
            if len(sft_item['question']) <= args.max_length and len(sft_item['answer']) <= args.max_length:
                filtered_data.append(item)
        
        dropped = len(all_data) - len(filtered_data)
        print(f"Dropped {dropped} samples due to length limit. Remaining: {len(filtered_data)}")
        all_data = filtered_data
    
    if args.max_samples:
        all_data = all_data[:args.max_samples]
        print(f"Using first {args.max_samples} samples")
    
    # 划分训练集和测试集
    split_idx = int(len(all_data) * args.train_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # 转换数据
    print("Converting to verl SFT format...")
    train_verl = [convert_message_to_sft_format(item) 
                  for item in tqdm(train_data, desc="Converting train")]
    test_verl = [convert_message_to_sft_format(item) 
                 for item in tqdm(test_data, desc="Converting test")]
    
    # 保存为parquet
    train_df = pd.DataFrame(train_verl)
    test_df = pd.DataFrame(test_verl)
    
    train_path = output_path / "train.parquet"
    test_path = output_path / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")
    
    # 保存示例JSON用于参考
    example_path = output_path / "train_example.json"
    with open(example_path, 'w', encoding='utf-8') as f:
        json.dump(train_verl[0], f, indent=2, ensure_ascii=False)
    print(f"Saved example to {example_path}")
    
    # 打印示例
    print("\n" + "="*50)
    print("Example converted data:")
    print("="*50)
    example = train_verl[0]
    print(f"question: {example['question'][:200]}...")
    print(f"answer: {example['answer'][:200]}...")

if __name__ == "__main__":
    main()
