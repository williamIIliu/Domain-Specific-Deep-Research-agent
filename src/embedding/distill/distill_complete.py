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
#         except Exception as e:
#             raise RuntimeError(f"角色检索失败：{str(e)}") from e

#         # 4. Stage1：生成角色、问题类型、难度
#         stage1_prompt = emd_stage1.format(
#             passage=passage_str,
#             characters=characters
#         )

async def call_llm_with_backoff(llm_client, model, messages, temperature=0.1, timeout=60, max_retries=5):
    """带指数退避的 LLM 调用封装"""
    for attempt in range(max_retries):
        try:
            resp = await llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                extra_body={"enable_thinking": False}
            )
            return resp
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower():
                wait_time = (2 ** attempt) + random.random()
                await asyncio.sleep(wait_time)
            elif attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                raise e
    return None

#         stage1_resp = await call_llm_with_backoff(
#             llm_client=llm_client,
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": stage1_prompt}
#             ],
#             temperature=0.1,
#             timeout=60,
#             max_retries=5
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
import asyncio
import time
import traceback
import random
import re
from typing import List, Dict
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv
import threading

from prompt import emd_stage1, emd_stage2
from build_persona_db import PersonaRetriever

# 配置参数
INPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean.jsonl"
OUTPUT_JSONL_PATH = "./datasets/embedding/querys.jsonl"
PERSONA_INDEX_DIR = "./datasets/persona-hub/finance_persona_index"
LLM_MODEL = "MiniMax-M2.5"
SYSTEM_PROMPT = "你是金融领域的专业分析助手"

# 并发配置
MAX_CONCURRENT_REQUESTS = 10  # 降低并发以适应 60 RPM 的限制
BATCH_SIZE = 100              # 异步处理不需要显式分小批次，这里仅用于进度展示

def load_jsonl(file_path: str, start: int, end: int) -> List[Dict]:
    """读取JSONL文件并返回文档列表"""
    docs = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                if not all(key in doc for key in ["id", "contents"]):
                    continue
                docs.append(doc)
            except json.JSONDecodeError:
                continue

    print(f"✅ 成功读取 {len(docs)} 个有效文档")
    return docs[start:end]

class AsyncThreadSafeWriter:
    """异步线程安全的文件写入器"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock = asyncio.Lock()
    
    async def write_doc(self, doc: Dict) -> None:
        async with self.lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")

def parse_json_robust(text):
    """鲁棒地解析 LLM 返回的 JSON，处理 markdown 代码块和 <think> 标签"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'```json\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    text = re.sub(r'```\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法解析 JSON 格式: {text[:200]}...")

async def process_single_doc_async(
    doc: Dict,
    persona_retriever: PersonaRetriever,
    llm_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    writer: AsyncThreadSafeWriter,
    pbar: tqdm
) -> None:
    """异步处理单个文档"""
    async with semaphore:
        try:
            # 1. 结构化数据抽样过滤
            raw_contents = doc.get("contents", "")
            if isinstance(raw_contents, dict) and random.random() < 0.7:
                print(f"ℹ️ 文档 {doc.get('id', 'unknown')[:8]} 因结构化抽样被跳过")
                pbar.update(1)
                return

            doc_contents = str(raw_contents).strip()
            if not doc_contents:
                print(f"⚠️ 文档 {doc.get('id', 'unknown')[:8]} 内容为空，跳过")
                pbar.update(1)
                return

            # 2. 检索候选角色
            loop = asyncio.get_event_loop()
            try:
                query_text = doc_contents[:256] if len(doc_contents) > 256 else doc_contents
                candidates = await loop.run_in_executor(
                    None, 
                    lambda: persona_retriever.retrieve_similar_personas(query_text=query_text, top_k=5)
                )

                if not isinstance(candidates, list) or not candidates:
                    print(f"⚠️ 文档 {doc['id'][:8]} 未匹配到角色，跳过")
                    pbar.update(1)
                    return

                valid_personas = [str(item["persona"]).strip() for item in candidates if isinstance(item, dict) and "persona" in item]
                if not valid_personas:
                    print(f"⚠️ 文档 {doc['id'][:8]} 无有效角色，跳过")
                    pbar.update(1)
                    return
                characters = "；".join(valid_personas)
            except Exception as e:
                print(f"❌ 文档 {doc['id'][:8]} 角色检索异常: {e}")
                pbar.update(1)
                return

            # 3. Stage1：生成角色、问题类型、难度
            stage1_prompt = emd_stage1.format(passage=json.dumps({"id": doc["id"], "contents": doc_contents}, ensure_ascii=False), characters=characters)
            
            stage1_result = None
            try:
                resp = await call_llm_with_backoff(
                    llm_client, 
                    LLM_MODEL, 
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": stage1_prompt}]
                )
                if resp:
                    resp_content = resp.choices[0].message.content
                    stage1_result = parse_json_robust(resp_content)
                    if not all(k in stage1_result for k in ["Characters", "Question_Type", "Difficulty"]):
                        print(f"⚠️ 文档 {doc['id'][:8]} Stage1 结果缺失字段: {list(stage1_result.keys())}")
                        stage1_result = None
            except Exception as e:
                print(f"❌ 文档 {doc['id'][:8]} Stage1 彻底失败: {e}")
                import traceback
                traceback.print_exc()

            if not stage1_result:
                pbar.update(1)
                return

            character = str(stage1_result.get("Characters", "")).strip()
            question_type = str(stage1_result.get("Question_Type", "")).strip()
            difficulty = str(stage1_result.get("Difficulty", "")).strip()

            if not character:
                print(f"⚠️ 文档 {doc['id'][:8]} Stage1 结果解析角色为空")
                pbar.update(1)
                return

            # 4. Stage2：生成Query
            stage2_prompt = emd_stage2.format(passage=doc_contents, character=character, type=question_type, difficulty=difficulty)
            
            final_query = ""
            fuzzy_query = ""
            try:
                resp = await call_llm_with_backoff(
                    llm_client,
                    LLM_MODEL,
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": stage2_prompt}]
                )
                if resp:
                    raw_content = resp.choices[0].message.content.strip()
                    # 彻底移除 think 标签内容
                    raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
                    
                    try:
                        # 尝试解析为 JSON 并提取 Query 和 Fuzzy_Query
                        res = parse_json_robust(raw_content)
                        if isinstance(res, dict):
                            # 优先尝试 Query，兼容旧的 Generated_Query
                            final_query = str(res.get("Query") or res.get("Generated_Query") or "").strip()
                            fuzzy_query = str(res.get("Fuzzy_Query") or "").strip()
                        else:
                            final_query = str(res).strip()
                    except Exception as e:
                        print(f"⚠️ 文档 {doc['id'][:8]} Stage2 JSON 解析失败，尝试回退到文本提取: {e}")
                        # 如果不是 JSON，则清理 Markdown 块后作为纯文本（回退方案）
                        final_query = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', raw_content, flags=re.DOTALL).strip()
            except Exception as e:
                print(f"❌ 文档 {doc['id'][:8]} Stage2 彻底失败: {e}")

            if not final_query:
                pbar.update(1)
                return

            # 5. 保存结果
            final_doc = {
                "id": doc["id"],
                "contents": doc_contents,
                **({"metadata": doc["metadata"]} if "metadata" in doc and isinstance(doc["metadata"], dict) else {}),
                "character": character,
                "question_type": question_type,
                "difficulty": difficulty,
                "query": final_query,
                "fuzzy_query": fuzzy_query
            }
            await writer.write_doc(final_doc)
            pbar.update(1)

        except Exception as e:
            print(f"❌ 处理文档 {doc.get('id', 'unknown')[:8]} 发生未捕获异常: {e}")
            pbar.update(1)

async def async_main(start, end):
    load_dotenv()
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_URL")
    
    llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    persona_retriever = PersonaRetriever(PERSONA_INDEX_DIR)
    writer = AsyncThreadSafeWriter(OUTPUT_JSONL_PATH)
    
    input_docs = load_jsonl(INPUT_JSONL_PATH, start, end)
    if not input_docs: return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    print(f"🚀 开始异步并发处理（共{len(input_docs)}个文档，最大并发：{MAX_CONCURRENT_REQUESTS}）")
    
    with tqdm(total=len(input_docs), desc="📊 处理进度") as pbar:
        tasks = [
            process_single_doc_async(doc, persona_retriever, llm_client, semaphore, writer, pbar)
            for doc in input_docs
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(async_main(args.start, args.end))

