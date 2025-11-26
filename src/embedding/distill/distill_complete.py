import json
import os
import time
import traceback
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

from prompt import emd_stage1, emd_stage2
from src.embedding.distill.build_persona_db import PersonaRetriever

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
    return docs[50010:50020]


def append_single_doc_to_jsonl(doc: Dict, file_path: str) -> None:
    """单个文档追加写入JSONL文件（不擦除原有数据）"""
    with open(file_path, "a", encoding="utf-8") as f:
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


def process_single_doc(
        doc: Dict,
        llm_client: OpenAI,
        persona_retriever: PersonaRetriever
) -> Dict:
    """处理单个文档，修复角色检索和格式错误"""
    try:
        # 1. 安全获取文档内容（防止contents为空或非字符串）
        doc_contents = str(doc.get("contents", "")).strip()
        if not doc_contents:
            raise ValueError("文档内容为空，无法处理")

        # 2. 预处理文档
        passage = {
            "id": doc["id"],
            "contents": doc_contents
        }
        passage_str = json.dumps(passage, ensure_ascii=False)

        # 3. 检索候选角色（核心修复：验证返回结果格式）
        try:
            # 限制query_text长度，避免超长文本导致检索错误
            query_text = doc_contents[:256] if len(doc_contents) > 256 else doc_contents
            candidates = persona_retriever.retrieve_similar_personas(
                query_text=query_text,  # 确保传入的是字符串而非切片
                top_k=5
            )

            # 验证candidates格式：必须是列表，且元素为含"persona"键的字典
            if not isinstance(candidates, list):
                raise TypeError(f"角色检索返回非列表类型：{type(candidates)}")

            valid_personas = []
            for idx, item in enumerate(candidates):
                if not isinstance(item, dict) or "persona" not in item:
                    print(f"⚠️ 过滤无效角色数据（第{idx + 1}个）：{str(item)[:50]}")
                    continue
                # 确保persona是字符串
                persona_str = str(item["persona"]).strip()
                if persona_str:
                    valid_personas.append(persona_str)

            if not valid_personas:
                raise ValueError("未获取到有效角色数据，无法继续处理")

            characters = "；".join(valid_personas)
            print(f"📌 有效角色数：{len(valid_personas)}")

        except Exception as e:
            raise RuntimeError(f"角色检索失败：{str(e)}") from e

        # 4. Stage1：生成角色、问题类型、难度
        stage1_prompt = emd_stage1.format(
            passage=passage_str,
            characters=characters
        )
        stage1_resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": stage1_prompt}
            ],
            extra_body={"enable_thinking": False},
            temperature=0.1,
            stream=False
        )

        # 验证Stage1返回是否为JSON
        stage1_content = stage1_resp.choices[0].message.content.strip()
        try:
            stage1_result = json.loads(stage1_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Stage1返回非JSON格式：{stage1_content[:100]}") from e

        # 提取Stage1结果（验证必要字段）
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
        stage2_resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": stage2_prompt}
            ],
            extra_body={"enable_thinking": False},
            temperature=0.1,
            stream=False
        )

        # 处理Stage2结果
        generated_query = stage2_resp.choices[0].message.content.strip()
        try:
            query_json = json.loads(generated_query)
            if isinstance(query_json, dict):
                final_query = str(query_json.get("Generated_Query", generated_query)).strip()
            else:
                # 如果解析结果是list或str，直接当成query
                final_query = str(query_json).strip()
        except json.JSONDecodeError:
            # 如果不是合法JSON，就直接原样使用
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
        final_doc["_process_status"] = "success"
        print(f"✅ 文档[{doc['id'][:8]}...]处理成功")
        return final_doc

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:50]}"
        print(f"❌ 文档[{doc['id'][:8]}...]处理失败：{error_msg}")
        return {"_process_status": "failed", "id": doc["id"]}


def main():
    try:
        print("=" * 60)
        print("🚀 开始文档串行处理流程（修复格式错误）")
        print("=" * 60)

        # 1. 初始化资源
        print("\n1. 初始化依赖资源")
        llm_client = init_llm_client()
        persona_retriever = PersonaRetriever(PERSONA_INDEX_DIR)
        print("✅ 所有依赖资源初始化完成")

        # 2. 读取输入文档
        print("\n2. 读取输入文档")
        input_docs = load_jsonl(INPUT_JSONL_PATH)
        if not input_docs:
            print("⚠️ 无有效文档，程序退出")
            return

        # 3. 串行处理文档
        print(f"\n3. 开始串行处理（共{len(input_docs)}个文档）")
        success_count = 0
        failed_count = 0

        for doc in tqdm(input_docs, desc="📊 串行处理进度"):
            result = process_single_doc(doc, llm_client, persona_retriever)
            if result["_process_status"] == "success":
                del result["_process_status"]
                append_single_doc_to_jsonl(result, OUTPUT_JSONL_PATH)
                success_count += 1
            else:
                failed_count += 1
            time.sleep(0.1)  # 避免API请求过于密集

        # 4. 输出统计
        print("\n4. 处理统计报告")
        total_count = len(input_docs)
        print(f"📋 统计结果：")
        print(f"   - 总处理文档数：{total_count}")
        print(f"   - 成功数：{success_count}（{round(success_count / total_count * 100, 1)}%）")
        print(f"   - 失败数：{failed_count}（{round(failed_count / total_count * 100, 1)}%）")
        print(f"   - 输出文件：{OUTPUT_JSONL_PATH}")

        print("\n" + "=" * 60)
        print("🎉 串行处理流程完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 程序全局异常：{str(e)}")
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    # 配置参数
    INPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean.jsonl"
    OUTPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean_query.jsonl"
    PERSONA_INDEX_DIR = "./datasets/persona-hub/finance_persona_index"
    LLM_MODEL = "qwen3-30b-a3b-instruct-2507"#"qwen3-30b-a3b"
    SYSTEM_PROMPT = "你是金融领域的专业分析助手"

    main()
    # file = load_jsonl("../../datasets/OmniEval-Corpus/all_data_clean.jsonl")
    # print(file)
