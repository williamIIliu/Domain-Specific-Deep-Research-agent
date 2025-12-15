import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from prompt_matrix import *
from topic_tree import topic_tree, topic_tree_hash, translate_topic_path
from task_tree import task_tree
import json_repair
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 OpenAI client（需在函数外部）
load_dotenv()
client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_URL"),
)

def format_multi_docs(sample):
    """
    生成包含主文档和不定数量相关文档的格式化字符串

    参数:
        main_doc (str): 主文档内容
        relevant_docs (list): 相关文档列表，每个元素是一个文档字符串

    返回:
        str: 按指定格式组合的完整字符串
    """
    main_doc = doc_str_format.format(title=sample["metadata"].get("Title", ""), content=sample["contents"])
    relevant_docs = sample["relevant_contents"]
    parts = [f"### 主文档\n{main_doc}\n"]

    # 循环添加相关文档（根据列表长度动态生成）
    for i, doc in enumerate(relevant_docs, start=0):
        doc = doc_str_format.format(title=doc["metadata"].get("Title", ""), content=doc["contents"])
        parts.append(f"### 相关文档 {i}\n{doc}\n")

    # 拼接所有部分，并用strip()去除首尾多余空行
    return ''.join(parts).strip()

def load_jsonl(file_path: str, start: int = 0, end: int | None = None):
    """读取JSONL文件并返回文档列表"""
    docs = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc="读取输入JSONL"), 0):
            if line_idx < start:
                continue
            if end is not None and line_idx >= end:
                break
            line_num = line_idx + 1
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                if not all(key in doc for key in ["id", "contents", "metadata", "relevant_contents"]):
                    print(f"⚠️ 跳过第{line_num}行: 缺少必需字段(id/contents/metadata/relevant_contents)")
                    continue
                docs.append(doc)
            except json.JSONDecodeError as e:
                print(f"⚠️ 跳过第{line_num}行: JSON解析错误 - {str(e)[:50]}")

    print(f"✅ 成功读取 {len(docs)} 个有效文档")
    return docs

def pipeline_demo(text_sample, candidate_workers: int = 1):
    """
    简化版 pipeline 流程
    1. topic 分类
    2. task 分类
    3. 数据生成
    4. 数据过滤
    """
    # filter the funding data
    if isinstance(text_sample["contents"], dict) and random.random() < 0.8:
        print("股票数据，已过滤")
        return

    # res
    res = [{}]
    res[0]["id"] = text_sample["id"]
    res[0]["contents"] = text_sample["contents"]
    res[0]["metadata"] = text_sample["metadata"]

    # ------------------ 1. Topic ------------------
    # print("Topic Configuration", topic_tree)
    topic_user_input = topic_classify_user.format(
        title=text_sample["metadata"].get("Title", " "),
        content=text_sample["contents"],
        topics_str=topic_tree
    )
    topic_messages = [
        {"role": "system", "content": topic_classify_system},
        {"role": "user", "content": topic_user_input}
    ]

    completion = client.chat.completions.create(
        model="qwen3-30b-a3b-instruct-2507",
        messages=topic_messages,
        temperature=0.1,
        extra_body={"enable_thinking": False},
        stream=False
    )
    topic_response = json.loads(completion.choices[0].message.content.strip())
    topic_id = int(topic_response["topic_id"])
    if topic_id == 0:
        print("Topic ID=0，流程终止")
        return None
    res[0]["topic"] = topic_tree_hash[topic_id]
    print("Approporate topic for this context:\n",topic_tree_hash[topic_id])
    # print("Current res after the topic:\n", res)

    # ------------------ 2. Task 分类 ------------------
    task_strs = []
    task_hash = []
    for i, (task_name, desc) in enumerate(task_tree.items()):
        try:
            desc = desc.split("### 任务要求")[1].strip()
        except IndexError:
            desc = "未找到明确任务要求"
        task_strs.append(json.dumps({"id": i, "name": task_name, "description": desc}, ensure_ascii=False))
        task_hash.append(task_name)
    # print("task configuration", task_strs, task_hash)

    # Combine the content and the relevant refence contents 
    doc_str = format_multi_docs(text_sample)
    # print("The context for distill process",doc_str)

    task_user_input = task_classify_user.format(
        doc_str=doc_str,
        task_str="\n".join(task_strs),
        topic_str=res[0]["topic"]
    )
    task_messages = [
        {"role": "system", "content": task_classify_system},
        {"role": "user", "content": task_user_input}
    ]

    completion = client.chat.completions.create(
        model="qwen-plus",#"qwen3-next-80b-a3b-thinking", #"qwen3-30b-a3b",
        messages=task_messages,
        extra_body={"enable_thinking": False},
        temperature=0.2,
        stream=False
    )
    task_response = json.loads(completion.choices[0].message.content.strip())
    # print(task_response)
    task_ids = task_response["task_id_list"]

    if not task_ids:  # 没有结果
        print("No suitable task for this content, break up.")
        return []

    new_res = []
    for r in res:
        for i, task_id in enumerate(task_ids):
            if task_id < len(task_hash):
                new_r = r.copy()  # 拷贝已有结果
                new_r["task"] = task_hash[task_id]

                new_r["relevant_contents_idxs"] = task_response["selected_relevant_contents_idx"][i]
                new_r["relevant_contents_ids"] = [text_sample["relevant_contents"][idx_content]["id"] for idx_content in task_response["selected_relevant_contents_idx"][i]]
                new_res.append(new_r)
            else:
                print(f"⚠️ task_id {task_id} 超出范围，忽略")
    res = new_res
    print("Approporate topic for this task:\n",task_ids)
    # print("Current res after the task:\n", res)

    # ------------------ 3. 数据生成 ------------------
    def generate_for_candidate(candidate):
        task_name = candidate["task"]

        main_doc = doc_str_format.format(
            title=candidate["metadata"].get("Title", ""),
            content=candidate["contents"]
        )
        parts = [f"### 文档\n{main_doc}\n"]

        relevant_node_ids = [candidate["id"]]

        for i, relevant_doc_idx in enumerate(candidate["relevant_contents_idxs"]):
            relevant_doc = text_sample["relevant_contents"][relevant_doc_idx]
            doc = doc_str_format.format(
                title=relevant_doc["metadata"].get("Title", ""),
                content=relevant_doc["contents"]
            )
            parts.append(f"### 相关文档 {i}\n{doc}\n")
            relevant_node_ids.append(relevant_doc["id"])

        doc_str = ''.join(parts).strip()
        print(f"\n{'='*50}\n任务类型: {task_name}\n{'='*50}")
        print(f"文档数量: 1 + {len(candidate['relevant_contents_idxs'])} 个相关文档")

        task_require = task_tree[task_name].replace("### 任务要求", "").strip()

        generation_user_input = data_generation_user.format(
            topic_name=candidate["topic"],
            task_name=task_name,
            task_require=task_require,
            doc_str=doc_str
        )

        generation_messages = [
            {"role": "system", "content": data_generation_system},
            {"role": "user", "content": generation_user_input}
        ]

        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=generation_messages,
                extra_body={"enable_thinking": False},
                temperature=0.2,
                stream=False
            )

            raw_response = completion.choices[0].message.content.strip()
            final_response = json_repair.loads(raw_response)

            if not isinstance(final_response, list):
                final_response = [final_response]

            print(f"生成数据数量: {len(final_response)}")

            generated_data = process_task_output(
                task_name=task_name,
                candidate=candidate,
                final_response=final_response,
                relevant_node_ids=relevant_node_ids
            )

            if generated_data:
                print(f"✅ 成功生成 {len(generated_data)} 条 {task_name} 数据")
                return generated_data
            print(f"⚠️ 任务 {task_name} 未生成有效数据")
            return []

        except Exception as e:
            print(f"❌ 解析或处理数据时发生错误: {str(e)}，跳过本轮处理")
            import traceback
            traceback.print_exc()
            return []

    res_generations = []
    if candidate_workers and candidate_workers > 1 and len(res) > 1:
        with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
            future_to_candidate = {executor.submit(generate_for_candidate, c): c for c in res}
            for future in as_completed(future_to_candidate):
                generated_data = future.result()
                if generated_data:
                    res_generations.extend(generated_data)
    else:
        for candidate in res:
            generated_data = generate_for_candidate(candidate)
            if generated_data:
                res_generations.extend(generated_data)

    return res_generations

def process_task_output(task_name, candidate, final_response, relevant_node_ids):
    """
    根据不同任务类型处理输出格式
    
    任务类型与输出格式对应关系：
    - 抽取类问答: 单条问答，answer为列表
    - 多跳推理类问答: 单条问答，需要多步推理
    - 对比类问答: 单条问答，涉及对比
    - 长答案形式问答: 单条问答，答案较长
    - 多轮对话能力: 多轮对话列表格式
    """
    results = []
    
    if not final_response or len(final_response) == 0:
        return results
    
    topic_name = candidate["topic"]
    
    if task_name == "多轮对话能力":
        # 多轮对话：整个response作为一个对话序列
        # 输出格式: [{"question": ..., "answer": ..., "relevant_passage": ...}, ...]
        conversation_turns = []
        for turn in final_response:
            if not validate_qa_item(turn):
                continue
            turn_data = {
                "question": turn["question"],
                "answer": ensure_list(turn["answer"]),
                "relevant_passage": ensure_list(turn.get("relevant_passage", [])),
                "topic_name": topic_name,
                "task_name": task_name,
                "relevant_node": relevant_node_ids
            }
            conversation_turns.append(turn_data)
        
        if len(conversation_turns) >= 2:  # 至少2轮才算多轮对话
            results.append(conversation_turns)
    else:
        # 其他任务类型：每条response独立处理
        for item in final_response:
            if not validate_qa_item(item):
                continue
            
            data_item = {
                "question": item["question"],
                "answer": ensure_list(item["answer"]),
                "relevant_passage": ensure_list(item.get("relevant_passage", [])),
                "topic_name": topic_name,
                "task_name": task_name,
                "relevant_node": relevant_node_ids if len(relevant_node_ids) > 1 else relevant_node_ids[0]
            }
            results.append(data_item)
    
    return results

def validate_qa_item(item):
    """验证问答项是否有效"""
    if not isinstance(item, dict):
        return False
    if "question" not in item or "answer" not in item:
        return False
    if not item["question"] or not item["answer"]:
        return False
    # 排除无效答案
    invalid_answers = ["无", "空", "无法回答", "无法根据检索文档回答问题"]
    answer = item["answer"]
    if isinstance(answer, str) and answer in invalid_answers:
        return False
    if isinstance(answer, list) and len(answer) == 1 and answer[0] in invalid_answers:
        return False
    return True

def ensure_list(value):
    """确保值为列表格式"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def get_task_filename(task_name):
    """根据任务名称获取对应的文件名"""
    task_file_mapping = {
        "多跳推理类问答": "multi-hop-reasoning.jsonl",
        "对比类问答": "contrastive.jsonl",
        "长答案形式问答": "long-form.jsonl",
        "多轮对话能力": "conversational.jsonl"
    }
    return task_file_mapping.get(task_name, "other.jsonl")

def save_generation_results(results, output_dir, topic_name=None):
    """
    保存生成结果到对应的文件
    
    Args:
        results: 生成的数据列表
        output_dir: 输出目录
        topic_name: 可选的主题名称，用于创建子目录
    """
    if not results:
        print("没有数据需要保存")
        return
    
    # 创建输出目录
    if topic_name:
        # 将中文主题名称转换为英文目录路径
        english_topic_path = translate_topic_path(topic_name)
        output_path = os.path.join(output_dir, english_topic_path)
    else:
        output_path = output_dir
    
    os.makedirs(output_path, exist_ok=True)
    
    # 按任务类型分组保存
    task_groups = {}
    for item in results:
        if isinstance(item, list):  # 多轮对话
            task_name = item[0]["task_name"] if item else "unknown"
        else:
            task_name = item.get("task_name", "unknown")
        
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(item)
    
    # 保存到对应文件
    for task_name, items in task_groups.items():
        filename = get_task_filename(task_name)
        filepath = os.path.join(output_path, filename)
        
        with open(filepath, "a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"📁 已保存 {len(items)} 条 {task_name} 数据到 {filepath}")

def run_batch_pipeline(input_jsonl_path, output_dir, max_docs=None, start: int = 0, end: int | None = None, candidate_workers: int = 1):
    """
    批量处理JSONL文件中的文档
    
    Args:
        input_jsonl_path: 输入JSONL文件路径
        output_dir: 输出目录
        max_docs: 最大处理文档数（可选）
    """
    files = load_jsonl(input_jsonl_path, start=start, end=end)
    
    if max_docs:
        files = files[:max_docs]
    
    total_generated = 0
    
    for i, file in enumerate(tqdm(files, desc="蒸馏处理进度", unit="doc")):
        doc_id = file.get("id", f"doc_{i}")
        try:
            results = pipeline_demo(file, candidate_workers=candidate_workers)
            
            if results and len(results) > 0:
                # 获取主题名称用于分类保存
                if isinstance(results[0], list):
                    topic_name = results[0][0].get("topic_name") if results[0] else None
                else:
                    topic_name = results[0].get("topic_name")
                
                save_generation_results(results, output_dir, topic_name)
                total_generated += len(results)
                print(f"✅ 文档 {doc_id}: 生成 {len(results)} 条数据")
            else:
                print(f"📄 文档 {doc_id}: 无有效生成数据")
                
        except Exception as e:
            print(f"❌ 文档 {doc_id}: 处理失败 - {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*50}")
    print(f"处理完成! 共处理 {len(files)} 个文档，生成 {total_generated} 条数据")
    print(f"{'='*50}")

# ------------------ 主程序入口 ------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="金融领域RAG评测数据蒸馏生成工具")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "batch"],
                        help="运行模式: demo(单文档测试) 或 batch(批量处理)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入JSONL文件路径 (batch模式必需)")
    parser.add_argument("--output", type=str, default="./output/rag_generation",
                        help="输出目录路径")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="最大处理文档数 (可选)")
    parser.add_argument("--candidate_workers", type=int, default=1,
                        help="task candidate并发数(>1启用并行)")
    parser.add_argument("--start", type=int, default=0,
                        help="蒸馏起始行号(0-based, inclusive)，batch模式有效")
    parser.add_argument("--end", type=int, default=None,
                        help="蒸馏截止行号(0-based, exclusive)，batch模式有效")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        # 单文档测试模式
        print("=" * 60)
        print("运行模式: 单文档测试 (demo)")
        print("=" * 60)
        
        text_sample = {
            "id": "b3ee734d-2417-42d1-b0b8-45ae35e92a28",
            "contents": "国家外汇管理局副局长、新闻发言人王春英就2022年上半年国际收支状况答记者问_管理资讯_青岛市分局 日前，国家外汇管理局公布了2022年二季度及上半年国际收支平衡表初步数据。国家外汇管理局副局长、新闻发言人王春英就相关问题回答了记者提问。 问：2022年上半年我国国际收支状况有何特点？ 答：国际收支平衡表初步数据显示，2022年上半年我国国际收支保持基本平衡。其中，经常账户顺差1691亿美元，与同期国内生产总值（gdp）之比为1.9%，继续处于合理均衡区间；直接投资净流入749亿美元，保持在较高水平。",
            "metadata": {"source_file": "190879.pkl", "Title": "2022年上半年国际收支状况"},
            "relevant_contents": [
                {
                    "id": "c538bfb2", 
                    "contents": "国家外汇管理局副局长、新闻发言人王春英就2022年上半年国际收支状况答记者问_管理资讯_广西壮族自治区分局\n日前，国家外汇管理局公布了2022年二季度及上半年国际收支平衡表初步数据。国家外汇管理局副局长、新闻发言人王春英就相关问题回答了记者提问。\n问：2022年上半年我国国际收支状况有何特点？\n答：国际收支平衡表初步数据显示，2022年上半年我国国际收支保持基本平衡。",
                    "metadata": {"source_file": "363002.pkl", "Title": "广西分局国际收支问答"}
                }
            ]
        }
        
        results = pipeline_demo(text_sample, candidate_workers=args.candidate_workers)
        
        if results:
            print(f"\n{'='*60}")
            print(f"生成结果预览 (共 {len(results)} 条):")
            print("=" * 60)
            for i, item in enumerate(results[:3]):  # 只显示前3条
                print(f"\n--- 第 {i+1} 条 ---")
                if isinstance(item, list):  # 多轮对话
                    print(f"类型: 多轮对话 ({len(item)} 轮)")
                    for j, turn in enumerate(item[:2]):  # 显示前2轮
                        print(f"  轮次{j+1} Q: {turn['question'][:50]}...")
                else:
                    print(f"类型: {item.get('task_name', 'unknown')}")
                    print(f"问题: {item['question'][:80]}...")
                    answer = item['answer'][0] if item['answer'] else ""
                    print(f"答案: {answer[:80]}...")
            
            # 保存结果
            os.makedirs(args.output, exist_ok=True)
            save_generation_results(results, args.output)
        else:
            print("未生成有效数据")
            
    elif args.mode == "batch":
        # 批量处理模式
        if not args.input:
            print("错误: batch模式需要指定 --input 参数")
            exit(1)
        
        print("=" * 60)
        print(f"运行模式: 批量处理 (batch)")
        print(f"输入文件: {args.input}")
        print(f"输出目录: {args.output}")
        if args.max_docs:
            print(f"最大文档数: {args.max_docs}")
        if args.candidate_workers and args.candidate_workers > 1:
            print(f"candidate并发数: {args.candidate_workers}")
        if args.start or args.end is not None:
            print(f"处理行范围: [{args.start}, {args.end})")
        print("=" * 60)
        
        run_batch_pipeline(args.input, args.output, args.max_docs, start=args.start, end=args.end, candidate_workers=args.candidate_workers)
