# import json
# import os
# import time
# import traceback
# from typing import List, Dict
# from tqdm import tqdm
# from openai import OpenAI, OpenAIError
# from dotenv import load_dotenv

# from prompt import emd_stage1, emd_stage2
# from build_persona_db import PersonaRetriever

# def load_jsonl(file_path: str) -> List[Dict]:
#     """读取JSONL文件并返回文档列表"""
#     docs = []
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"输入文件不存在: {file_path}")

#     with open(file_path, "r", encoding="utf-8") as f:
#         for line_num, line in enumerate(tqdm(f, desc="读取输入JSONL"), 1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 doc = json.loads(line)
#                 if not all(key in doc for key in ["id", "contents"]):
#                     print(f"⚠️ 跳过第{line_num}行: 缺少id/contents字段")
#                     continue
#                 docs.append(doc)
#             except json.JSONDecodeError as e:
#                 print(f"⚠️ 跳过第{line_num}行: JSON解析错误 - {str(e)[:50]}")

#     print(f"✅ 成功读取 {len(docs)} 个有效文档")
#     # return docs[50010:50020]
#     return docs


# def append_single_doc_to_jsonl(doc: Dict, file_path: str) -> None:
#     """单个文档追加写入JSONL文件（不擦除原有数据）"""
#     with open(file_path, "a", encoding="utf-8") as f:
#         json.dump(doc, f, ensure_ascii=False)
#         f.write("\n")


# def init_llm_client() -> OpenAI:
#     """初始化LLM客户端"""
#     load_dotenv()
#     api_key = os.getenv("QWEN_API_KEY")
#     base_url = os.getenv("QWEN_URL")

#     if not api_key or not base_url:
#         raise ValueError("❌ 请在.env文件中配置QWEN_API_KEY和QWEN_URL")

#     try:
#         return OpenAI(api_key=api_key, base_url=base_url)
#     except Exception as e:
#         raise RuntimeError(f"❌ LLM客户端初始化失败：{str(e)}") from e


# def process_single_doc(
#         doc: Dict,
#         llm_client: OpenAI,
#         persona_retriever: PersonaRetriever
# ) -> Dict:
#     """处理单个文档，修复角色检索和格式错误"""
#     try:
#         # 1. 安全获取文档内容（防止contents为空或非字符串）
#         doc_contents = str(doc.get("contents", "")).strip()
#         if not doc_contents:
#             raise ValueError("文档内容为空，无法处理")

#         # 2. 预处理文档
#         passage = {
#             "id": doc["id"],
#             "contents": doc_contents
#         }
#         passage_str = json.dumps(passage, ensure_ascii=False)

#         # 3. 检索候选角色（核心修复：验证返回结果格式）
#         try:
#             # 限制query_text长度，避免超长文本导致检索错误
#             query_text = doc_contents[:256] if len(doc_contents) > 256 else doc_contents
#             candidates = persona_retriever.retrieve_similar_personas(
#                 query_text=query_text,  # 确保传入的是字符串而非切片
#                 top_k=5
#             )

#             # 验证candidates格式：必须是列表，且元素为含"persona"键的字典
#             if not isinstance(candidates, list):
#                 raise TypeError(f"角色检索返回非列表类型：{type(candidates)}")

#             valid_personas = []
#             for idx, item in enumerate(candidates):
#                 if not isinstance(item, dict) or "persona" not in item:
#                     print(f"⚠️ 过滤无效角色数据（第{idx + 1}个）：{str(item)[:50]}")
#                     continue
#                 # 确保persona是字符串
#                 persona_str = str(item["persona"]).strip()
#                 if persona_str:
#                     valid_personas.append(persona_str)

#             if not valid_personas:
#                 raise ValueError("未获取到有效角色数据，无法继续处理")

#             characters = "；".join(valid_personas)
#             print(f"📌 有效角色数：{len(valid_personas)}")

#         except Exception as e:
#             raise RuntimeError(f"角色检索失败：{str(e)}") from e

#         # 4. Stage1：生成角色、问题类型、难度
#         stage1_prompt = emd_stage1.format(
#             passage=passage_str,
#             characters=characters
#         )
#         stage1_resp = llm_client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": stage1_prompt}
#             ],
#             extra_body={"enable_thinking": False},
#             temperature=0.1,
#             stream=False
#         )

#         # 验证Stage1返回是否为JSON
#         stage1_content = stage1_resp.choices[0].message.content.strip()
#         try:
#             stage1_result = json.loads(stage1_content)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Stage1返回非JSON格式：{stage1_content[:100]}") from e

#         # 提取Stage1结果（验证必要字段）
#         required_fields = ["Characters", "Question_Type", "Difficulty"]
#         for field in required_fields:
#             if field not in stage1_result:
#                 raise KeyError(f"Stage1结果缺少必要字段：{field}")

#         character = str(stage1_result["Characters"]).strip()
#         question_type = str(stage1_result["Question_Type"]).strip()
#         difficulty = str(stage1_result["Difficulty"]).strip()

#         # 5. Stage2：生成Query
#         stage2_prompt = emd_stage2.format(
#             passage=doc_contents,
#             character=character,
#             type=question_type,
#             difficulty=difficulty
#         )
#         stage2_resp = llm_client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": stage2_prompt}
#             ],
#             extra_body={"enable_thinking": False},
#             temperature=0.1,
#             stream=False
#         )

#         # 处理Stage2结果
#         generated_query = stage2_resp.choices[0].message.content.strip()
#         try:
#             query_json = json.loads(generated_query)
#             if isinstance(query_json, dict):
#                 final_query = str(query_json.get("Generated_Query", generated_query)).strip()
#             else:
#                 # 如果解析结果是list或str，直接当成query
#                 final_query = str(query_json).strip()
#         except json.JSONDecodeError:
#             # 如果不是合法JSON，就直接原样使用
#             final_query = generated_query.strip()

#         # 构建最终结果
#         final_doc = {
#             "id": doc["id"],
#             "contents": doc_contents,
#             **({"metadata": doc["metadata"]} if "metadata" in doc and isinstance(doc["metadata"], dict) else {}),
#             "character": character,
#             "question_type": question_type,
#             "difficulty": difficulty,
#             "query": final_query
#         }
#         final_doc["_process_status"] = "success"
#         print(f"✅ 文档[{doc['id'][:8]}...]处理成功")
#         return final_doc

#     except Exception as e:
#         error_msg = f"{type(e).__name__}: {str(e)[:50]}"
#         print(f"❌ 文档[{doc['id'][:8]}...]处理失败：{error_msg}")
#         return {"_process_status": "failed", "id": doc["id"]}


# def main():
#     try:
#         print("=" * 60)
#         print("🚀 开始文档串行处理流程（修复格式错误）")
#         print("=" * 60)

#         # 1. 初始化资源
#         print("\n1. 初始化依赖资源")
#         llm_client = init_llm_client()
#         persona_retriever = PersonaRetriever(PERSONA_INDEX_DIR)
#         print("✅ 所有依赖资源初始化完成")

#         # 2. 读取输入文档
#         print("\n2. 读取输入文档")
#         input_docs = load_jsonl(INPUT_JSONL_PATH)
#         if not input_docs:
#             print("⚠️ 无有效文档，程序退出")
#             return

#         # 3. 串行处理文档
#         print(f"\n3. 开始串行处理（共{len(input_docs)}个文档）")
#         success_count = 0
#         failed_count = 0

#         for doc in tqdm(input_docs, desc="📊 串行处理进度"):
#             result = process_single_doc(doc, llm_client, persona_retriever)
#             if result["_process_status"] == "success":
#                 del result["_process_status"]
#                 append_single_doc_to_jsonl(result, OUTPUT_JSONL_PATH)
#                 success_count += 1
#             else:
#                 failed_count += 1
#             time.sleep(0.1)  # 避免API请求过于密集

#         # 4. 输出统计
#         print("\n4. 处理统计报告")
#         total_count = len(input_docs)
#         print(f"📋 统计结果：")
#         print(f"   - 总处理文档数：{total_count}")
#         print(f"   - 成功数：{success_count}（{round(success_count / total_count * 100, 1)}%）")
#         print(f"   - 失败数：{failed_count}（{round(failed_count / total_count * 100, 1)}%）")
#         print(f"   - 输出文件：{OUTPUT_JSONL_PATH}")

#         print("\n" + "=" * 60)
#         print("🎉 串行处理流程完成")
#         print("=" * 60)

#     except Exception as e:
#         print(f"\n❌ 程序全局异常：{str(e)}")
#         traceback.print_exc()
#         print("=" * 60)


# if __name__ == "__main__":
#     # 配置参数
#     INPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean_new2.jsonl"
#     OUTPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean_query.jsonl"
#     PERSONA_INDEX_DIR = "./datasets/persona-hub/finance_persona_index"
#     LLM_MODEL = "qwen3-30b-a3b-instruct-2507"#"qwen3-30b-a3b"
#     # LLM_MODEL = "qwen3-30b-a3b"
#     SYSTEM_PROMPT = "你是金融领域的专业分析助手"

#     main()
#     # file = load_jsonl("../../datasets/OmniEval-Corpus/all_data_clean.jsonl")
#     # print(file)




import json
import os
import time
import traceback
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import copy

from prompt import emd_stage1, emd_stage2
from build_persona_db import PersonaRetriever

def load_jsonl(file_path: str) -> List[Dict]:
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
                if not all(key in doc for key in ["id", "contents"]):
                    print(f"⚠️ 跳过第{line_num}行: 缺少id/contents字段")
                    continue
                docs.append(doc)
            except json.JSONDecodeError as e:
                print(f"⚠️ 跳过第{line_num}行: JSON解析错误 - {str(e)[:50]}")

    print(f"✅ 成功读取 {len(docs)} 个有效文档")
    return docs


class ThreadSafeWriter:
    """线程安全的文件写入器"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock = threading.Lock()
    
    def write_doc(self, doc: Dict) -> None:
        """线程安全地写入单个文档"""
        with self.lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")


def init_llm_client() -> OpenAI:
    """初始化LLM客户端"""
    load_dotenv()
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_URL")

    if not api_key or not base_url:
        raise ValueError("❌ 请在.env文件中配置QWEN_API_KEY和QWEN_URL")

    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        raise RuntimeError(f"❌ LLM客户端初始化失败：{str(e)}") from e


def process_single_doc_concurrent(
    doc: Dict,
    persona_retriever: PersonaRetriever,
    thread_id: int
) -> Dict:
    """并发处理单个文档 - 每个线程创建自己的LLM客户端"""
    try:
        # 每个线程创建自己的LLM客户端，避免并发冲突
        llm_client = init_llm_client()
        
        # 1. 安全获取文档内容
        doc_contents = str(doc.get("contents", "")).strip()
        if not doc_contents:
            raise ValueError("文档内容为空，无法处理")

        # 2. 预处理文档
        passage = {
            "id": doc["id"],
            "contents": doc_contents
        }
        passage_str = json.dumps(passage, ensure_ascii=False)

        # 3. 检索候选角色
        try:
            query_text = doc_contents[:256] if len(doc_contents) > 256 else doc_contents
            candidates = persona_retriever.retrieve_similar_personas(
                query_text=query_text,
                top_k=5
            )

            if not isinstance(candidates, list):
                raise TypeError(f"角色检索返回非列表类型：{type(candidates)}")

            valid_personas = []
            for idx, item in enumerate(candidates):
                if not isinstance(item, dict) or "persona" not in item:
                    continue
                persona_str = str(item["persona"]).strip()
                if persona_str:
                    valid_personas.append(persona_str)

            if not valid_personas:
                raise ValueError("未获取到有效角色数据，无法继续处理")

            characters = "；".join(valid_personas)

        except Exception as e:
            raise RuntimeError(f"角色检索失败：{str(e)}") from e

        # 4. Stage1：生成角色、问题类型、难度
        stage1_prompt = emd_stage1.format(
            passage=passage_str,
            characters=characters
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stage1_resp = llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": stage1_prompt}
                    ],
                    extra_body={"enable_thinking": False},
                    temperature=0.1,
                    stream=False,
                    timeout=30  # 添加超时
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # 重试前等待

        # 验证Stage1返回
        stage1_content = stage1_resp.choices[0].message.content.strip()
        try:
            stage1_result = json.loads(stage1_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Stage1返回非JSON格式：{stage1_content[:100]}") from e

        # 提取Stage1结果
        required_fields = ["Characters", "Question_Type", "Difficulty"]
        for field in required_fields:
            if field not in stage1_result:
                raise KeyError(f"Stage1结果缺少必要字段：{field}")

        character = str(stage1_result["Characters"]).strip()
        question_type = str(stage1_result["Question_Type"]).strip()
        difficulty = str(stage1_result["Difficulty"]).strip()

        # 5. Stage2：生成Query
        stage2_prompt = emd_stage2.format(
            passage=doc_contents,
            character=character,
            type=question_type,
            difficulty=difficulty
        )
        
        for attempt in range(max_retries):
            try:
                stage2_resp = llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": stage2_prompt}
                    ],
                    extra_body={"enable_thinking": False},
                    temperature=0.1,
                    stream=False,
                    timeout=30
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)

        # 处理Stage2结果
        generated_query = stage2_resp.choices[0].message.content.strip()
        try:
            query_json = json.loads(generated_query)
            if isinstance(query_json, dict):
                final_query = str(query_json.get("Generated_Query", generated_query)).strip()
            else:
                final_query = str(query_json).strip()
        except json.JSONDecodeError:
            final_query = generated_query.strip()

        # 构建最终结果
        final_doc = {
            "id": doc["id"],
            "contents": doc_contents,
            **({"metadata": doc["metadata"]} if "metadata" in doc and isinstance(doc["metadata"], dict) else {}),
            "character": character,
            "question_type": question_type,
            "difficulty": difficulty,
            "query": final_query
        }
        
        print(f"✅ [线程{thread_id}] 文档[{doc['id'][:8]}...]处理成功")
        return {"status": "success", "data": final_doc}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
        print(f"❌ [线程{thread_id}] 文档[{doc['id'][:8]}...]处理失败：{error_msg}")
        return {"status": "failed", "id": doc["id"], "error": error_msg}


def process_batch_concurrent(args):
    """处理一批文档的包装函数"""
    docs_batch, persona_retriever, thread_id = args
    results = []
    
    for doc in docs_batch:
        result = process_single_doc_concurrent(doc, persona_retriever, thread_id)
        results.append(result)
        # 添加小延迟，避免API请求过于密集
        time.sleep(0.05)
    
    return results


def main():
    try:
        print("=" * 60)
        print("🚀 开始文档并发处理流程")
        print("=" * 60)

        # 1. 初始化资源
        print("\n1. 初始化依赖资源")
        persona_retriever = PersonaRetriever(PERSONA_INDEX_DIR)
        
        # 初始化线程安全的写入器
        writer = ThreadSafeWriter(OUTPUT_JSONL_PATH)
        print("✅ 所有依赖资源初始化完成")

        # 2. 读取输入文档
        print("\n2. 读取输入文档")
        input_docs = load_jsonl(INPUT_JSONL_PATH)
        if not input_docs:
            print("⚠️ 无有效文档，程序退出")
            return

        # 3. 并发处理配置
        max_workers = MAX_WORKERS  # 并发线程数
        batch_size = BATCH_SIZE    # 每批处理的文档数
        
        print(f"\n3. 开始并发处理（共{len(input_docs)}个文档）")
        print(f"📊 并发配置: {max_workers}个线程，每批{batch_size}个文档")

        # 4. 分批处理
        success_count = 0
        failed_count = 0
        
        # 将文档分批
        batches = []
        for i in range(0, len(input_docs), batch_size):
            batch = input_docs[i:i + batch_size]
            batches.append((batch, persona_retriever, i // batch_size))

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(process_batch_concurrent, batch_args): batch_idx 
                for batch_idx, batch_args in enumerate(batches)
            }
            
            # 处理完成的任务
            with tqdm(total=len(input_docs), desc="📊 并发处理进度") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        
                        # 处理批次结果
                        for result in batch_results:
                            if result["status"] == "success":
                                writer.write_doc(result["data"])
                                success_count += 1
                            else:
                                failed_count += 1
                            pbar.update(1)
                            
                    except Exception as e:
                        print(f"❌ 批次 {batch_idx} 处理异常: {str(e)}")
                        # 批次失败，所有文档都标记为失败
                        batch_size_actual = len(batches[batch_idx][0])
                        failed_count += batch_size_actual
                        pbar.update(batch_size_actual)

        # 5. 输出统计
        print("\n4. 处理统计报告")
        total_count = len(input_docs)
        print(f"📋 统计结果：")
        print(f"   - 总处理文档数：{total_count}")
        print(f"   - 成功数：{success_count}（{round(success_count / total_count * 100, 1)}%）")
        print(f"   - 失败数：{failed_count}（{round(failed_count / total_count * 100, 1)}%）")
        print(f"   - 输出文件：{OUTPUT_JSONL_PATH}")

        print("\n" + "=" * 60)
        print("🎉 并发处理流程完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 程序全局异常：{str(e)}")
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    # 配置参数
    INPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean_new2.jsonl"
    OUTPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean_query_concurrent.jsonl"
    PERSONA_INDEX_DIR = "./datasets/persona-hub/finance_persona_index"
    LLM_MODEL = "qwen3-30b-a3b-instruct-2507"
    SYSTEM_PROMPT = "你是金融领域的专业分析助手"
    
    # 并发配置 - 根据你的API限制和服务器性能调整
    MAX_WORKERS = 8      # 并发线程数，建议从4-8开始测试
    BATCH_SIZE = 10      # 每批处理的文档数，建议10-50

    main()
