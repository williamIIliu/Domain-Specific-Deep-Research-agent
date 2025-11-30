"""
PRM Reward Score 计算模块
支持三种奖励类型：
1. 格式奖励 (format_reward) - 检查输出是否符合 <think>...</think><answer>...</answer> 格式
2. 答案奖励 (answer_reward) - 检查最终答案是否正确
3. 过程奖励 (process_reward) - 使用 PRM 模型评估推理过程质量
"""

import re
import os
from typing import List, Dict, Optional, Tuple
import requests
from dotenv import load_dotenv

load_dotenv()

# 尝试导入 math_verify，如果不存在则使用简单的字符串比较
try:
    from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


# ============================================================
# 配置
# ============================================================
PRM_API_URL = os.getenv("REWARD_URL", "http://0.0.0.0:8060/v1/chat/completions")
PRM_API_KEY = os.getenv("REWARD_API_KEY")
PRM_MODEL_NAME = "reward_model"

# 奖励权重配置
DEFAULT_WEIGHTS = {
    "format": 0.5,    # 格式奖励权重
    "answer": 1.0,    # 答案奖励权重
    "process": 1.0,   # 过程奖励权重
}


# ============================================================
# 格式提取函数
# ============================================================
def extract_think(text: str) -> Optional[str]:
    """提取 <think>...</think> 中的内容"""
    if not isinstance(text, str):
        return None
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if len(matches) >= 1 else None


def extract_steps(think_content: str) -> List[Dict]:
    """
    从思考内容中提取步骤
    严格格式: 只支持大写 Step (Step 1:, Step1:, Step 1：等)
    
    Returns:
        List[Dict]: [{"step_num": 1, "content": "..."}, ...]
    """
    if not isinstance(think_content, str):
        return []
    
    steps = []
    
    # 严格匹配大写 Step 格式
    # Step 1: 或 Step1: 或 Step 1： (中文冒号)
    pattern = r"Step\s*(\d+)\s*[：:]\s*(.+?)(?=Step\s*\d+|$)"
    matches = re.findall(pattern, think_content, re.DOTALL)
    
    if matches:
        for step_num, content in matches:
            steps.append({
                "step_num": int(step_num),
                "content": content.strip()
            })
    
    return steps


def validate_steps_format(think_content: str) -> Tuple[bool, int]:
    """
    验证思考内容是否包含有效的步骤格式
    严格要求: 必须使用大写 Step 格式
    
    Returns:
        (is_valid, step_count): 是否有效，步骤数量
    """
    steps = extract_steps(think_content)
    
    # 必须有步骤且步骤内容有效
    if len(steps) == 0:
        return False, 0
    
    # 检查步骤是否有实质内容（至少4个字符）
    valid_steps = [s for s in steps if len(s["content"]) > 3]
    
    return len(valid_steps) > 0, len(valid_steps)


def extract_answer(text: str) -> Optional[str]:
    """提取 <answer>...</answer> 中的内容"""
    if not isinstance(text, str):
        return None
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if len(matches) >= 1 else None


def extract_box_content(text: str) -> Optional[str]:
    """
    提取答案框中的内容
    严格格式: 只支持 Qwen3 格式 <|box_start|>...<|box_end|>
    """
    if not isinstance(text, str):
        return None
    
    # 只匹配 Qwen3 格式: <|box_start|>...<|box_end|>
    pattern_qwen = r"<\|box_start\|>(.*?)<\|box_end\|>"
    matches = re.findall(pattern_qwen, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    return None


# ============================================================
# 格式奖励
# ============================================================
def compute_format_reward(response: str) -> Tuple[float, Dict]:
    """
    计算格式奖励
    检查输出是否符合 <think>...</think><answer>...<|box_start|>...<|box_end|></answer> 格式
    同时检查思考过程是否包含有效的步骤格式
    
    Returns:
        (score, details): 分数 [0, 1] 和详细信息
    """
    details = {
        "has_think": False,
        "has_answer": False,
        "has_box": False,
        "has_steps": False,
        "step_count": 0,
    }
    
    think_content = extract_think(response)
    answer_content = extract_answer(response)
    box_content = extract_box_content(response)
    
    details["has_think"] = think_content is not None and len(think_content.strip()) > 0
    details["has_answer"] = answer_content is not None
    details["has_box"] = box_content is not None
    
    # 检查步骤格式
    if think_content:
        has_steps, step_count = validate_steps_format(think_content)
        details["has_steps"] = has_steps
        details["step_count"] = step_count
    
    # 计算分数：
    # - think 标签: 0.2
    # - 步骤格式: 0.2
    # - answer 标签: 0.3
    # - box 内容: 0.3
    score = sum([
        0.20 if details["has_think"] else 0,
        0.20 if details["has_steps"] else 0,
        0.30 if details["has_answer"] else 0,
        0.30 if details["has_box"] else 0,
    ])
    
    # 如果 step_count 大于 5，格式奖励为 0
    if details["step_count"] > 5:
        score = 0.0
    
    return score, details


# ============================================================
# 答案奖励
# ============================================================
def normalize_answer(answer: str) -> List[str]:
    """标准化答案，返回可能的等价形式列表"""
    if answer is None:
        return []
    
    answer = str(answer).strip()
    results = [answer]
    
    if HAS_MATH_VERIFY:
        try:
            results.extend(parse(answer, extraction_mode="first_match"))
            results.extend(parse(answer, extraction_config=[StringExtractionConfig()]))
            results.extend(parse(answer, extraction_config=[LatexExtractionConfig()]))
        except Exception:
            pass
    
    return list(set(results))


def compute_answer_reward(response: str, ground_truth: str) -> Tuple[float, Dict]:
    """
    计算答案奖励
    比较模型输出的答案与标准答案
    
    Returns:
        (score, details): 分数 [0, 1] 和详细信息
    """
    details = {
        "extracted_answer": None,
        "ground_truth": ground_truth,
        "is_correct": False,
    }
    
    # 提取模型答案
    answer_content = extract_answer(response)
    if answer_content:
        extracted = extract_box_content(answer_content)
        if extracted is None:
            extracted = extract_box_content(response)
        if extracted is None:
            extracted = answer_content.strip()
        details["extracted_answer"] = extracted
    else:
        # 尝试直接从 response 提取 box
        extracted = extract_box_content(response)
        if extracted:
            details["extracted_answer"] = extracted
    
    if details["extracted_answer"] is None:
        return 0.0, details
    
    # 标准化并比较
    pred_answers = normalize_answer(details["extracted_answer"])
    true_answers = normalize_answer(ground_truth)
    
    if HAS_MATH_VERIFY:
        try:
            is_correct = verify(pred_answers, true_answers)
        except Exception:
            is_correct = any(p == t for p in pred_answers for t in true_answers)
    else:
        is_correct = any(p == t for p in pred_answers for t in true_answers)
    
    details["is_correct"] = is_correct
    return 1.0 if is_correct else 0.0, details


# ============================================================
# 过程奖励 (PRM)
# ============================================================
def compute_process_reward(
    query: str, 
    response: str,
    model_answer: str,
    ground_truth: str,
    api_url: str = PRM_API_URL,
    api_key: str = PRM_API_KEY,
    model_name: str = PRM_MODEL_NAME,
) -> Tuple[float, Dict]:
    """
    使用 PRM 模型计算过程奖励
    
    Returns:
        (score, details): 分数归一化到 [0, 1]，和详细信息
    """
    details = {
        "raw_score": None,
        "error": None,
    }
    
    think_content = extract_think(response)
    if think_content is None:
        details["error"] = "No think content found"
        return 0.0, details
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 构造评估 prompt
        eval_prompt = f"""请作为金融领域的专家，评估以下推理过程的质量，给出 0-10 的分数。

问题：{query}

标准答案：
{ground_truth}

模型生成的推理过程：
{think_content}

模型生成的最终答案：
{model_answer}

评分标准（用于评估思维/推理过程的质量，而不是只看最终答案）：
1. 推理过程的一致性：
   - 各步骤之间是否逻辑连贯，上下文是否前后一致，没有自相矛盾。
2. 逐步正确性：
   - 使用的公式是否正确，数据代入是否正确，每一步计算是否存在明显算术错误。
3. 关键要素覆盖度：
   - 是否完整覆盖了解决该金融问题所必须的关键步骤（读取题干数据、选取合适金融公式/方法、代入计算、检查结果合理性等）。
4. 金融业务合理性：
   - 推理过程是否符合基本金融常识和约束（如金额符号、比例范围、时间维度、利率含义等），没有明显违背业务常识的推理。
5. 与标准答案的一致性：
   - 在不直接抄袭标准答案的前提下，思维过程是否能够合理推导出标准答案 {ground_truth}，或者至少朝着正确方向逐步逼近。

请综合以上维度给出一个 0-10 的总评分（0 表示推理过程几乎完全错误或无关，10 表示推理过程非常清晰、严谨且能够正确推导出标准答案）。
只输出一个数字分数（0-10），不要输出任何其他文字。"""
        
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
            details["error"] = f"API error: {resp.status_code}"
            
    except Exception as e:
        details["error"] = str(e)
    
    return 0.0, details


# ============================================================
# 综合评分函数
# ============================================================
def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict = None,
    weights: Dict[str, float] = None,
    use_process_reward: bool = False,
) -> float:
    """
    计算综合奖励分数
    
    Args:
        solution_str: 模型输出的完整响应
        ground_truth: 标准答案
        extra_info: 额外信息字典，包含 "query" 键用于过程奖励
        weights: 各奖励权重，默认 {"format": 0.1, "answer": 0.5, "process": 0.4}
        use_process_reward: 是否使用过程奖励（需要 PRM 服务）
        format_score: 仅格式正确时的基础分
        answer_score: 答案正确时的满分
    
    Returns:
        float: 综合奖励分数
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    else:
        weights = weights.copy()
    
    # 从 extra_info 中获取 query 和 ground_truth
    query = ""
    ground_truth_raw = ground_truth
    if extra_info and isinstance(extra_info, dict):
        query = extra_info.get("question", "")
        ground_truth_raw = extra_info.get("answer", ground_truth)
    
    # 1. 格式奖励
    format_reward, format_details = compute_format_reward(solution_str)
    
    # 2. 答案奖励
    answer_reward, answer_details = compute_answer_reward(solution_str, ground_truth)
    
    # 3. 过程奖励
    if use_process_reward and query:
        # 提取模型的最终答案
        model_answer = answer_details.get("extracted_answer", "")
        if not model_answer:
             model_answer = extract_answer(solution_str) or ""
             
        process_reward, _ = compute_process_reward(
            query=query, 
            response=solution_str, 
            model_answer=model_answer,
            ground_truth=ground_truth_raw
        )
    else:
        process_reward = 0.0
        weights["process"] = 0.0
        total = weights["format"] + weights["answer"]
        if total > 0:
            weights["format"] /= total
            weights["answer"] /= total
    
    # 加权求和
    final_score = (
        weights["format"] * format_reward +
        weights["answer"] * answer_reward +
        weights["process"] * process_reward
    )
    reward_extra_info = {
        "format": format_details,
        "answer": answer_details,
        "process": process_details,
        "weights": weights,
    }
    
    return {
        "reward_tensor": final_score,
        "reward_extra_info": reward_extra_info,
    }
    # return final_score

# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试用例 - 严格格式
    response = "<think>Step1: 分析问题\nStep2: 计算结果\n</think><answer>the final result is<|box_start|>A<|box_end|></answer>"
    ground_truth = "A"
    extra_info = {
        "question": "What is the answer?",
        "answer": "The step-by-step solution leads to A. #### A"
    }
    
    print("=" * 60)
    print("PRM Reward 测试 (严格格式)")
    print("=" * 60)
    print(f"\n测试响应: {response}")
    print(f"标准答案: {ground_truth}")
    
    # 测试格式奖励
    format_score, format_details = compute_format_reward(response)
    print(f"\n[格式奖励] 分数: {format_score:.2f}")
    print(f"  - has_think: {format_details['has_think']}")
    print(f"  - has_steps: {format_details['has_steps']} (必须大写 Step)")
    print(f"  - step_count: {format_details['step_count']}")
    print(f"  - has_answer: {format_details['has_answer']}")
    print(f"  - has_box: {format_details['has_box']} (只支持 Qwen3 格式)")
    
    # 测试答案奖励
    answer_score, answer_details = compute_answer_reward(response, ground_truth)
    print(f"\n[答案奖励] 分数: {answer_score:.2f}")
    print(f"  - extracted_answer: {answer_details['extracted_answer']}")
    print(f"  - ground_truth: {answer_details['ground_truth']}")
    print(f"  - is_correct: {answer_details['is_correct']}")
    
    # 测试综合评分（使用 extra_info 传递 query）
    final_score = compute_score(response, ground_truth, extra_info=extra_info, use_process_reward=False)
    print(f"\n[综合评分] 分数: {final_score:.2f} (无过程奖励)")
    
    # 测试带详情的综合评分
    final_score, details = compute_score_with_details(response, ground_truth, extra_info=extra_info, use_process_reward=False)
    print(f"\n[详细评分]")
    print(f"  - 最终分数: {details['final_score']:.2f}")
    print(f"  - 权重: format={details['weights']['format']:.2f}, answer={details['weights']['answer']:.2f}")
    print(f"  - 格式分: {details['format']['score']:.2f}")
    print(f"  - 答案分: {details['answer']['score']:.2f}")
    
    # 测试小写 step（应该不通过）
    print("\n" + "=" * 60)
    print("测试小写 step（应该不通过）")
    print("=" * 60)
    lowercase_response = "<think>step1: 分析\nstep2: 计算</think><answer><|box_start|>A<|box_end|></answer>"
    lowercase_score, lowercase_details = compute_format_reward(lowercase_response)
    print(f"响应: {lowercase_response}")
    print(f"格式分数: {lowercase_score:.2f}")
    print(f"  - has_steps: {lowercase_details['has_steps']} (小写 step 不通过)")
    
    # 测试 LaTeX boxed 格式（应该不通过）
    print("\n" + "=" * 60)
    print("测试 LaTeX \\boxed{} 格式（应该不通过）")
    print("=" * 60)
    latex_response = r"<think>Step1: 计算</think><answer>答案是 $\boxed{42}$</answer>"
    latex_score, latex_details = compute_format_reward(latex_response)
    print(f"响应: {latex_response}")
    print(f"格式分数: {latex_score:.2f}")
    print(f"  - has_box: {latex_details['has_box']} (LaTeX 格式不支持)")
    box = extract_box_content(latex_response)
    print(f"  - extracted_box: {box}")
    
    # 测试正确的 Qwen3 格式
    print("\n" + "=" * 60)
    print("测试正确的 Qwen3 格式")
    print("=" * 60)
    qwen3_response = "<think>Step1: 分析问题\nStep2: 计算 6*7=42\nStep3: 验证结果</think><answer>最终答案是<|box_start|>42<|box_end|></answer>"
    qwen3_score, qwen3_details = compute_format_reward(qwen3_response)
    print(f"响应: {qwen3_response}")
    print(f"格式分数: {qwen3_score:.2f}")
    print(f"  - has_think: {qwen3_details['has_think']}")
    print(f"  - has_steps: {qwen3_details['has_steps']}")
    print(f"  - step_count: {qwen3_details['step_count']}")
    print(f"  - has_answer: {qwen3_details['has_answer']}")
    print(f"  - has_box: {qwen3_details['has_box']}")

