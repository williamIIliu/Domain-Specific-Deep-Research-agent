# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import os
import random
import re
import string
from typing import Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 配置
# ============================================================
REWARD_API_URL = os.getenv("REWARD_URL", "http://0.0.0.0:8060/v1/chat/completions")
REWARD_API_KEY = os.getenv("REWARD_API_KEY")
REWARD_MODEL_NAME = "reward_model"

# 奖励权重配置
DEFAULT_WEIGHTS = {
    "format": 0.3,      # 格式奖励权重
    "answer": 0.7,      # 答案匹配奖励权重（使用 reward model 判断）
}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


# ============================================================
# 格式奖励
# ============================================================
def compute_format_reward(solution_str: str) -> Tuple[float, Dict]:
    """
    计算格式奖励
    检查输出是否包含有效的 <answer>...</answer> 格式
    
    Returns:
        (score, details): 分数 [0, 1] 和详细信息
    """
    details = {
        "has_answer": False,
        "answer_count": 0,
        "extracted_answer": None,
    }
    
    answer = extract_solution(solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    
    details["has_answer"] = answer is not None
    details["answer_count"] = open_count
    details["extracted_answer"] = answer
    
    # 计算分数
    if answer is None:
        return 0.0, details
    
    # 如果 answer 标签过多，扣分
    if open_count > 10 or close_count > 10:
        return 0.25, details
    
    return 1.0, details


# ============================================================
# 答案匹配奖励 (使用 Reward Model)
# ============================================================
def compute_answer_reward(
    question: str,
    extracted_answer: str,
    ground_truth_target: str,
    api_url: str = REWARD_API_URL,
    api_key: str = REWARD_API_KEY,
    model_name: str = REWARD_MODEL_NAME,
) -> Tuple[float, Dict]:
    """
    使用 Reward Model 计算答案匹配奖励
    判断模型答案与标准答案是否一致（替代传统的 EM 检查）
    
    Returns:
        (score, details): 分数归一化到 [0, 1]，和详细信息
    """
    details = {
        "raw_score": None,
        "error": None,
    }
    
    if extracted_answer is None or not extracted_answer.strip():
        details["error"] = "No answer extracted"
        return 0.0, details
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 构造答案匹配评估 prompt
        eval_prompt = f"""请判断以下两个答案是否一致，给出 0-10 的分数。

问题：{question}

标准答案：{ground_truth_target}

模型答案：{extracted_answer}

评分标准：
1. 如果两个答案完全相同或语义完全等价，给 10 分。
2. 如果两个答案表达的核心意思相同，但表述方式不同（如数字格式、单位表达等），给 7-9 分。
3. 如果模型答案部分正确，包含了标准答案的关键信息，给 4-6 分。
4. 如果模型答案与标准答案几乎无关或完全错误，给 0-3 分。

请只输出一个数字分数（0-10），不要输出任何其他文字。"""
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": eval_prompt}
            ],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if resp.status_code == 200:
            result = resp.json()
            score_text = result["choices"][0]["message"]["content"].strip()
            numbers = re.findall(r"(\d+(?:\.\d+)?)", score_text)
            if numbers:
                raw_score = float(numbers[0])
                details["raw_score"] = raw_score
                return min(raw_score / 10.0, 1.0), details
        else:
            details["error"] = f"API error: {resp.status_code}, body: {resp.text[:200]}"
            
    except Exception as e:
        details["error"] = str(e)
    
    return 0.0, details


# ============================================================
# 综合评分函数
# ============================================================
def compute_score(
    solution_str: str,
    ground_truth: Dict,
    extra_info: Dict = None,
    weights: Dict[str, float] = None,
    **kwargs,
) -> Dict:
    """
    计算综合奖励分数（兼容 PRM reward 格式）
    使用 Reward Model 替代传统的 EM 检查
    
    Args:
        solution_str: 模型输出的完整响应
        ground_truth: 标准答案字典，包含 "target" 键
        extra_info: 额外信息字典，包含 "question" 键
        weights: 各奖励权重
    
    Returns:
        Dict: 包含 score, format, answer, weights 的字典
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    else:
        weights = weights.copy()
    
    # 从 extra_info 中获取 question
    question = ""
    if extra_info and isinstance(extra_info, dict):
        question = extra_info.get("question", "")
    
    # 获取 target
    target = ground_truth.get("target", ground_truth) if isinstance(ground_truth, dict) else str(ground_truth)
    if isinstance(target, list):
        target = target[0] if target else ""
    
    # 1. 格式奖励
    format_reward, format_details = compute_format_reward(solution_str)
    extracted_answer = format_details.get("extracted_answer")
    
    # 2. 答案匹配奖励（使用 Reward Model 替代 EM）
    if extracted_answer and question:
        answer_reward, answer_details = compute_answer_reward(
            question=question,
            extracted_answer=extracted_answer,
            ground_truth_target=str(target),
        )
    else:
        answer_reward = 0.0
        answer_details = {"error": "No answer or question"}
    
    # 加权求和
    final_score = (
        weights["format"] * format_reward +
        weights["answer"] * answer_reward
    )
    
    # Debug 打印
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {target}")
        print(f"Extracted answer: {extracted_answer}")
        print(f"Format reward: {format_reward:.2f}, Answer reward: {answer_reward:.2f}")
        print(f"Final score: {final_score:.2f}")
        if answer_details.get("error"):
            print(f"Answer error: {answer_details['error']}")
    
    return {
        "score": final_score,
        "format": weights["format"] * format_reward,
        "answer": weights["answer"] * answer_reward,
        "weights": weights,
    }


# 保留原有函数名作为别名，保持向后兼容
compute_score_subem = compute_score
