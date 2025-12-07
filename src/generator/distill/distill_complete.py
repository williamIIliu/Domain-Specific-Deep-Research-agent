import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from prompt_matrix import *
from topic_tree import topic_tree, topic_tree_hash
from task_tree import task_tree
import json_repair
from tqdm import tqdm
import random

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

def load_jsonl(file_path: str) :
    """读取JSONL文件并返回文档列表"""
    docs = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="读取输入JSONL"), 1):
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

def pipeline_demo(text_sample):
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
    res_generations = []
    for candidate in res:
        # print(candidate)
        main_doc = doc_str_format.format(title=candidate["metadata"].get("Title", ""), content=candidate["contents"])

        parts = [f"### 文档\n{main_doc}\n"]
        # print(main_doc)

        # 循环添加相关文档（根据列表长度动态生成）
        for i, relevant_doc_idx in enumerate(candidate["relevant_contents_idxs"]):
            doc = doc_str_format.format(title=text_sample["relevant_contents"][relevant_doc_idx]["metadata"].get("Title", ""),
                                        content=text_sample["relevant_contents"][relevant_doc_idx]["contents"])
            parts.append(f"### 相关文档 {i}\n{doc}\n")

        # 拼接所有部分，并用strip()去除首尾多余空行
        doc_str =  ''.join(parts).strip()
        print(doc_str)

        generation_user_input = data_generation_user.format(
            topic_name=candidate["topic"],
            task_name=candidate["task"],
            task_require=task_tree[candidate["task"]].replace("### 任务要求", "").strip(),
            doc_str=doc_str
        )
        print("Data generation 环节生成用户prompt", generation_user_input)

        generation_messages = [
            {"role": "system", "content": data_generation_system},
            {"role": "user", "content": generation_user_input}
        ]

        completion = client.chat.completions.create(
            model="qwen-plus",  # "qwen3-next-80b-a3b-thinking", #"qwen3-30b-a3b",
            messages=generation_messages,
            extra_body={"enable_thinking": False},
            temperature=0.2,
            stream=False
        )

        final_response=json_repair.loads(completion.choices[0].message.content.strip())
        print("生成数据回复", final_response)

        try:

            if len(final_response) != 0:
                for cur in final_response:
                    candidate["thought_process"] = cur["thought_process"]
                    candidate["question"] = cur["question"]
                    candidate["answer"] = cur["answer"]
                    candidate["relevant_passage"] = cur["relevant_passage"]
                    print("一条高质量数据\n", candidate)
                    res_generations.append(candidate)
        except Exception as e:
            print(f"解析或处理数据时发生错误: {str(e)}，跳过本轮处理")

    # generation_user_input = data_generation_user.format(
    #     topic_name=passage["topic"],
    #     task_name=passage["task_type"],
    #     task_require=task_tree[passage["task_type"]],
    #     doc_str=doc_str_format.format(title=passage["metadata"].get("Title", ""), content=passage["contents"])
    # )
    # generation_messages = [
    #     {"role": "system", "content": data_generation_system},
    #     {"role": "user", "content": generation_user_input}
    # ]
    
    # completion = client.chat.completions.create(
    #     model="qwen3-30b-a3b",
    #     messages=generation_messages,
    #     temperature=0.1,
    #     stream=False
    # )
    # final_response = json_repair.loads(completion.choices[0].message.content.strip())
    # final_response = final_response if isinstance(final_response, list) else [final_response]
    
    # print("【测试】生成结果数量：", len(final_response))

    # # ------------------ 4. 数据规范化输出 ------------------
    # res["question"] = final_response[0]["question"]
    # res["answer"] = final_response[0]["answer"]
    # res["relevant_passage"] = final_response[0].get("relevant_passage", "")
    # res["relevant_contents_ID"] = [passage, relevant_content]
    # res["relevant_contents"] = [passage, relevant_content]
    # res["topic_name"] = passage["topic"]
    # res["task_name"] = passage["task_type"]
    #
    # print("【最终结果】", json.dumps(res, ensure_ascii=False, indent=2))
    # return res_generations


# ------------------ 测试 ------------------
if __name__ == "__main__":
    # 示例文档
    text_sample = {
        "id": "b3ee734d-2417-42d1-b0b8-45ae35e92a28",
        "contents": " 国家外汇管理局副局长、新闻发言人王春英就2022年上半年国际收支状况答记者问_管理资讯_青岛市分局 日前，国家外汇管理局公布了2022年二季度及上半年国际收支平衡表初步数据。国家外汇管理局副局长、新闻发言人王春英就相关问题回答了记者提问。 问：2022年上半年我国国际收支状况有何特点？ 答：国际收支平衡表初步数据显示，2022年上半年我国国际收支保持基本平衡。其中，经常账户顺差1691亿美元，与同期国内生产总值（gdp）之比为1.9%，继续处于合理均衡区间；直接投资净流入749亿美元，保持在较高水平。 一是货物贸易顺差同比增长。2022年上半年，我国货物贸易进出口呈现较强的韧性。我国国际收支口径的货物贸易顺差3207亿美元，增长36%，为历年同期最高值。其中，货物贸易出口16437亿美元，同比增长13%；进口13230亿美元，同比增长8%。 二是服务贸易逆差同比收窄。2022年上半年，服务贸易逆差378亿美元，同比下降30%。其中，旅行逆差519亿美元，同比增长31%，主要是海外留学等支出有所回升；智慧财产权使用费逆差159亿美元，与2021年同期基本持平，收入和支出均有所增长，反映我国在智慧财产权领域国际合作不断扩大；运输逆差22亿美元，同比下降89%，主要是运输收入增速快于支出；电信、计算机和资讯服务顺差91亿美元，同比增长1.2倍，体现服务业数字化转型为我国服务贸易发展注入新动能。 三是直接投资保持较高水平净流入。2022年上半年，直接投资净流入749亿美元。其中，来华直接投资净流入1496亿美元，显示我国市场对外资保持吸引力；对外直接投资净流出747亿美元，总体平稳有序。 总的来看，我国高效统筹疫情防控和经济社会发展，经济韧性强、潜力大、活力足，长期向好的基本面没有改变，有利于我国国际收支继续保持基本平衡。",
        "metadata":{"source_file": "190879.pkl"},
        "relevant_contents": [
            {"id": "c538bfb2", "contents": "国家外汇管理局副局长、新闻发言人王春英就2022年上半年国际收支状况答记者问_管理资讯_广西壮族自治区分局\n日前，国家外汇管理局公布了2022年二季度及上半年国际收支平衡表初步数据。国家外汇管理局副局长、新闻发言人王春英就相关问题回答了记者提问。\n问：2022年上半年我国国际收支状况有何特点？\n答：国际收支平衡表初步数据显示，2022年上半年我国国际收支保持基本平衡。其中，经常账户顺差1691亿美元，与同期国内生产总值（GDP）之比为1.9%，继续处于合理均衡区间；直接投资净流入749亿美元，保持在较高水平。\n一是货物贸易顺差同比增长。2022年上半年，我国货物贸易进出口呈现较强的韧性。我国国际收支口径的货物贸易顺差3207亿美元，增长36%，为历年同期最高值。其中，货物贸易出口16437亿美元，同比增长13%；进口13230亿美元，同比增长8%。\n二是服务贸易逆差同比收窄。2022年上半年，服务贸易逆差378亿美元，同比下降30%。其中，旅行逆差519亿美元，同比增长31%，主要是海外留学等支出有所回升；智慧财产权使用费逆差159亿美元，与2021年同期基本持平，收入和支出均有所增长，反映我国在智慧财产权领",
             "metadata": {"source_file": "363002.pkl"}
             }
        ]
    }
    result = pipeline_demo(text_sample)

    # INPUTPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/RAG_test/RAG_base_data.jsonl"
    # OUTPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/RAG_test/RAG_generation_data.jsonl"

    # files = load_jsonl(INPUTPUT_JSONL_PATH)
    # # 【修改：添加try-except捕获单个文档的所有错误】
    # for file in tqdm(files, desc="蒸馏处理进度", unit="doc"):
    #     doc_id = file.get("id", "未知ID")
    #     try:
    #         # 处理文档（若内部出错，直接进入except）
    #         result = pipeline_demo(file)

    #         # 原有逻辑：判断result是否合法并写入
    #         if isinstance(result, list) and len(result) > 0:
    #             with open(OUTPUT_JSONL_PATH, "a", encoding="utf-8") as f:
    #                 json.dump(result, f, ensure_ascii=False)
    #                 f.write("\n")
    #             print(f"✅ 文档ID {doc_id}: 写入 {len(result)} 条数据")
    #         else:
    #             print(f"📄 文档ID {doc_id}: 无有效生成数据，跳过写入")

    #     # 【新增：捕获该文档处理过程中的所有错误，跳过并继续】
    #     except Exception as e:
    #         # 打印详细错误信息（便于调试），但不中断程序
    #         print(f"\n❌ 文档ID {doc_id}: 处理失败，跳过该文档。错误：{str(e)[:100]}")
    #         # 可选：打印错误堆栈（需要导入traceback）
    #         # import traceback
    #         # traceback.print_exc()
    #         continue  # 跳过当前文档，处理下一个




