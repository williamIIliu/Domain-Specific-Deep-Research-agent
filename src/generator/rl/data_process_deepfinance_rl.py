import argparse
import os
import re
from typing import Any, Dict, List, Optional

import datasets

from verl.utils.hdfs_io import copy, makedirs


def strip_think_tags(text: str) -> str:
    """移除 <think>...</think> 包裹的思考内容，只保留最终回答部分。"""
    if not isinstance(text, str):
        return ""
    # 删除成对的 <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_last_non_empty_line(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines()]
    for ln in reversed(lines):
        if ln:
            return ln
    return ""


def detect_choice_answer(line: str) -> Optional[str]:
    """从一行文本中抽取单个选项答案（A/B/C/D/...）。"""
    if not line:
        return None
    # 只保留大写字母 A-Z
    candidates = re.findall(r"[A-Z]", line)
    if len(candidates) == 1:
        return candidates[0]
    # 常见格式如 "答案：C" 或 "Correct answer is C"
    m = re.search(r"答案[:：]?\s*([A-Z])", line)
    if m:
        return m.group(1)
    return None


def detect_numeric_answer(line: str) -> Optional[str]:
    """从一行文本中抽取数值答案（保留原字符串）。"""
    if not line:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", line.replace(",", ""))
    if not m:
        return None
    return m.group(0)


def is_choice_question(question: str) -> bool:
    """简单启发：包含 A:/B:/C: 即认为是选择题。"""
    if not isinstance(question, str):
        return False
    return all(x in question for x in ["A:", "B:", "C:"])


def classify_and_extract_ground_truth(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 Agentar DeepFinance 的一条 SFT 样本中抽取 RL 所需字段。

    只保留选择题（带选项 A/B/C/...），答案为选项字母。
    """
    msgs: List[Dict[str, str]] = example.get("messages", [])
    if len(msgs) < 2:
        return None

    user_msg = msgs[0].get("content", "")
    assistant_msg = msgs[1].get("content", "")
    if not user_msg or not assistant_msg:
        return None

    question_raw = user_msg.strip()

    # 先去掉 <think> 段，只看最终回答
    ans_body = strip_think_tags(assistant_msg)
    last_line = extract_last_non_empty_line(ans_body)

    # 只保留选择题
    if not is_choice_question(question_raw):
        return None
    
    choice = detect_choice_answer(last_line)
    if choice is None:
        # 再从整个回答里找一次
        choice = detect_choice_answer(ans_body)
    if choice is not None:
        return {
            "task_type": "mc_single",
            "question_raw": question_raw,
            "ground_truth": choice,
            "answer_raw": assistant_msg,
        }

    return None


def build_instruction_suffix(task_type: str) -> str:
    """Return a unified instruction template, independent of task_type."""
    return (
        """Solve the following question step by step (no more than 5 steps).  You must wrap your thinking with <think>Step1: ...\nStep2: ...\n</think>,  write the final answer between <answer> and </answer>,  and put the final result inside <|box_start|>result<|box_end|>."""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw JSONL dataset.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/agentar_deepfinance_rl",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--max_question_length",
        type=int,
        default=2048,
        help="Maximum question length. Questions longer than this will be filtered out.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    if local_dataset_path is None:
        local_dataset_path = "datasets/Agentar-DeepFinance-100K/Agentar_DeepFinance_sft.jsonl"

    # 使用 datasets 从 JSONL 加载
    dataset = datasets.load_dataset("json", data_files={"train": local_dataset_path})
    full_train = dataset["train"]

    # 过滤：只保留选择题，且问题长度不超过 max_question_length
    max_len = args.max_question_length
    
    def filter_fn(example):
        parsed = classify_and_extract_ground_truth(example)
        if parsed is None:
            return False
        # 过滤超长问题
        question_raw = parsed["question_raw"]
        if len(question_raw) > max_len:
            return False
        return True

    filtered_train = full_train.filter(filter_fn)
    print(f"Filtered dataset size: {len(filtered_train)} (from {len(full_train)}, max_len={max_len}, only_choice=True)")

    # 再对过滤后的数据进行转换
    def map_fn(example, idx):
        parsed = classify_and_extract_ground_truth(example)
        task_type = parsed["task_type"]
        question_raw = parsed["question_raw"]
        ground_truth = parsed["ground_truth"]
        answer_raw = parsed["answer_raw"]

        instruction = build_instruction_suffix(task_type)
        question = question_raw + instruction

        data = {
            "data_source": "prm_reward",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "finance",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "index": idx,
                "task_type": task_type,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data

    mapped = filtered_train.map(map_fn, with_indices=True, remove_columns=filtered_train.column_names)

    # 查看处理后的数据集样例
    print(f"Mapped dataset size: {len(mapped)}")
    print(mapped[0:3])

    # 按 98/2 划分 train/test
    split_dataset = mapped.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    os.makedirs(os.path.expanduser(local_save_dir), exist_ok=True)
    local_save_dir = os.path.expanduser(local_save_dir)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
