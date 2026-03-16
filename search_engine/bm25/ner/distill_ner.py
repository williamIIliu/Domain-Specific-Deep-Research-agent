import json
import os
import asyncio
import re
import random
from typing import List, Dict
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import the prompt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "ner"))
from prompt import ner_system_prompt, ner_user_prompt_template

# Configuration
BASE_DIR = "/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent"
INPUT_JSONL_PATH = os.path.join(BASE_DIR, "datasets/OmniEval-Corpus/all_data_clean.jsonl")
OUTPUT_JSONL_PATH = os.path.join(BASE_DIR, "datasets/NER/FiNER_Eval.jsonl")
LLM_MODEL = "MiniMax-M2.5"
MAX_CONCURRENT_REQUESTS = 10
LIMIT = 10000

def load_jsonl(file_path: str, start_index: int, limit: int) -> List[Dict]:
    docs = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num < start_index:
                continue
            if len(docs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                if "contents" in doc:
                    docs.append(doc)
            except json.JSONDecodeError:
                continue
    print(f"✅ Successfully loaded {len(docs)} documents starting from index {start_index}")
    return docs

class AsyncThreadSafeWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.lock = asyncio.Lock()
    
    async def write_doc(self, doc: Dict) -> None:
        async with self.lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")

def parse_json_robust(text):
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
        raise ValueError(f"Could not parse JSON: {text[:200]}...")

async def call_llm_with_backoff(llm_client, model, messages, temperature=0.1, timeout=60, max_retries=5):
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

async def process_single_doc_async(
    doc: Dict,
    doc_index: int,
    llm_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    writer: AsyncThreadSafeWriter,
    pbar: tqdm,
    stats: Dict
) -> bool:
    """异步处理单个文档，返回是否成功生成 NER 数据"""
    async with semaphore:
        try:
            if stats["success_count"] >= stats["limit"]:
                return False

            raw_contents = doc.get("contents", "")
            
            # 1. 结构化数据抽样过滤
            if isinstance(raw_contents, dict) and random.random() < 0.7:
                stats["last_index"] = max(stats["last_index"], doc_index)
                pbar.update(1)
                return False

            content = str(raw_contents).strip()
            if not content:
                stats["last_index"] = max(stats["last_index"], doc_index)
                pbar.update(1)
                return False

            # Format user prompt
            user_prompt = ner_user_prompt_template.format(user_input_text=content)
            
            # Call LLM
            resp = await call_llm_with_backoff(
                llm_client,
                LLM_MODEL,
                [
                    {"role": "system", "content": ner_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            if resp:
                raw_output = resp.choices[0].message.content.strip()
                # 彻底移除 think 标签内容
                raw_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
                try:
                    # Validate JSON output
                    parse_json_robust(raw_output)
                    
                    # Construct ChatML format
                    chatml_doc = {
                        "messages": [
                            {"role": "system", "content": ner_system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": raw_output}
                        ]
                    }
                    await writer.write_doc(chatml_doc)
                    stats["success_count"] += 1
                    stats["last_index"] = max(stats["last_index"], doc_index)
                    pbar.update(1)
                    return True
                except Exception as e:
                    print(f"❌ Failed to parse output for doc {doc.get('id', 'unknown')}: {e}")
            
            stats["last_index"] = max(stats["last_index"], doc_index)
            pbar.update(1)
            return False
        except Exception as e:
            print(f"❌ Error processing doc {doc.get('id', 'unknown')}: {e}")
            stats["last_index"] = max(stats["last_index"], doc_index)
            pbar.update(1)
            return False

async def main(start_index: int, limit: int):
    load_dotenv()
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_URL")
    
    llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    writer = AsyncThreadSafeWriter(OUTPUT_JSONL_PATH)
    
    # 我们需要加载足够多的文档以确保能蒸馏出 limit 条成功的数据
    # 由于有过滤和可能的解析失败，我们加载比 limit 更多的文档
    load_limit = limit * 5  # 预估 20% 的成功率，可以根据实际调整
    input_docs_with_indices = []
    
    if not os.path.exists(INPUT_JSONL_PATH):
        print(f"Input file not found: {INPUT_JSONL_PATH}")
        return

    print(f"🔍 Searching for documents starting from index {start_index}...")
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue
            if len(input_docs_with_indices) >= load_limit:
                break
            try:
                doc = json.loads(line.strip())
                if "contents" in doc:
                    input_docs_with_indices.append((doc, idx))
            except:
                continue

    if not input_docs_with_indices:
        print("No documents to process.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stats = {
        "success_count": 0,
        "limit": limit,
        "last_index": start_index
    }
    
    print(f"🚀 Starting NER distillation (Target: {limit}, Max Concurrent: {MAX_CONCURRENT_REQUESTS})")
    
    with tqdm(total=limit, desc="📊 Progress (Success Count)") as pbar:
        # 逐个提交任务，并在达到限制时停止
        pending_tasks = set()
        doc_iter = iter(input_docs_with_indices)
        
        while stats["success_count"] < limit:
            # 补充任务到并发上限
            while len(pending_tasks) < MAX_CONCURRENT_REQUESTS and stats["success_count"] < limit:
                try:
                    doc, idx = next(doc_iter)
                    task = asyncio.create_task(process_single_doc_async(doc, idx, llm_client, semaphore, writer, pbar, stats))
                    pending_tasks.add(task)
                except StopIteration:
                    break
            
            if not pending_tasks:
                break
                
            # 等待至少一个任务完成
            done, pending_tasks = await asyncio.wait(
                pending_tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 检查是否有新成功的
            if stats["success_count"] >= limit:
                # 取消所有还在运行的任务
                for t in pending_tasks:
                    t.cancel()
                print(f"\n✅ Reached target of {limit} successful distillations.")
                break

    print("\n" + "="*50)
    print(f"🎉 Distillation Finished!")
    print(f"📈 Total Successful: {stats['success_count']}")
    print(f"📍 Last Processed Index: {stats['last_index']}")
    print(f"💡 To resume, use: --start {stats['last_index'] + 1}")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index in the corpus")
    parser.add_argument("--limit", type=int, default=10000, help="Number of records to process")
    args = parser.parse_args()
    
    asyncio.run(main(args.start, args.limit))
