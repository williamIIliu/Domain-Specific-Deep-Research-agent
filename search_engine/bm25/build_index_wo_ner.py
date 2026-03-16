import os
import json
import re
import subprocess
from tqdm import tqdm

def clean_finance_text(raw_content):
    """
    清洗金融数据：去除符号噪音，保留核心文本，对关键字段加权。
    """
    if isinstance(raw_content, dict):
        text_parts = []
        for k, v in raw_content.items():
            if v is not None:
                # 提取 Key 和 Value，用空格分隔
                text_parts.append(str(k))
                text_parts.append(str(v))
                # 针对核心字段（代码、名称、简称）进行加权：重复存入
                if any(key in str(k) for key in ["代码", "简称", "名称"]):
                    text_parts.append(str(v))
        text = " ".join(text_parts)
    elif isinstance(raw_content, list):
        text = " ".join([str(i) for i in raw_content])
    else:
        text = str(raw_content)

    # 关键步骤：去除 JSON 特殊符号 [ ] { } " : , 替换为空格
    clean_text = re.sub(r'["\'{}:,\[\]]', ' ', text)
    # 压缩连续空格
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

def build_BM25_index(jsonl_path, json_slice_path, index_path):
    # 创建目录
    os.makedirs(json_slice_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)
    
    cnt = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="正在清洗语料并生成片段"):
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            cnt += 1
            docid = obj.get("id", f"doc_{cnt}")
            
            # 1. 提取内容并清洗
            raw_content = obj.get("contents") or obj.get("text_chunk") or ""
            processed_content = clean_finance_text(raw_content)
            
            # 2. 提取元数据标题
            metadata = obj.get("metadata", {})
            title = metadata.get("Title", "") if isinstance(metadata, dict) else ""
            
            # 合并标题与内容
            final_text = f"{title} {processed_content}".strip()
            
            # 写入临时文件供 Pyserini 读取
            with open(os.path.join(json_slice_path, f"{docid}.json"), "w", encoding="utf-8") as fout:
                json.dump({"id": docid, "contents": final_text}, fout, ensure_ascii=False)

    print(f"\n预处理完成，共计 {cnt} 条。正在构建 Lucene 索引...")

    # 3. 调用 Pyserini (注意：不使用 --pretokenized，使用 --language zh)
    indexing_args = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", json_slice_path,
        "--index", index_path,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "16",
        "--language", "zh", 
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    subprocess.run(indexing_args, check=True)
    print(f"✅ 索引构建成功: {index_path}")

if __name__ == "__main__":
    # 配置路径
    JSONL_PATH = "datasets/OmniEval-Corpus/all_data_clean.jsonl"
    SLICE_DIR = "./datasets/database/bm25_slices_native"
    INDEX_DIR = "./datasets/database/bm25_index_native"
    
    build_BM25_index(JSONL_PATH, SLICE_DIR, INDEX_DIR)