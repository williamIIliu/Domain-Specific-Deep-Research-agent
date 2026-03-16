import os
import json
import numpy as np
import faiss
from tqdm import tqdm
import subprocess
import re
try:
    import jieba
except ImportError:
    jieba = None

# ------------------------- 文本读取 -------------------------
def load_corpus(corpus_path: str):
    corpus= []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                # 解析单行JSON
                data = json.loads(line)

                # 提取必要字段
                item = {
                    "id": data.get("id", ""),  # 确保id存在，无则为空字符串
                    "contents": "",
                    "embedding": None  # 预留Embedding字段，暂为None
                }

                # 处理contents字段
                contents = data.get("contents", "")
                if isinstance(contents, dict):
                    # 若为字典，转换为 "key":"value" 格式的字符串（用逗号分隔）
                    item["contents"] = ", ".join([f'"{k}":"{v}"' for k, v in contents.items()])
                else:
                    # 若为字符串，直接保留
                    item["contents"] = str(contents)  # 确保是字符串类型

            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败 - {str(e)}")
            except Exception as e:
                print(f"警告：第{line_num}行处理失败 - {str(e)}")

    return corpus

# ------------------------- BM25 Index -------------------------
def build_BM25_index(jsonl_path, json_slice_path, index_path, language="zh", ner_dict_path=None):
    # 创建目录
    os.makedirs(json_slice_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)

    # 加载自定义词典
    if ner_dict_path and os.path.exists(ner_dict_path):
        if jieba:
            print(f"Loading NER dictionary from: {ner_dict_path}")
            jieba.load_userdict(ner_dict_path)
        else:
            print("Warning: jieba not installed, cannot use custom dictionary for tokenization.")
    
    # 拆分JSONL为单个文件
    cnt = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="正在预处理语料"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cnt += 1
            docid = obj.get("id", f"doc_{cnt}")
            
            # 1. 提取核心内容
            raw_content = obj.get("contents") or obj.get("text_chunk") or obj.get("context") or ""
            
            # 2. 提取元数据 (金融数据中文件名和标题通常包含关键领域信息)
            metadata = obj.get("metadata", {})
            meta_text = ""
            if isinstance(metadata, dict):
                # 优先提取文件名和标题
                important_meta = []
                if "file_name" in metadata: important_meta.append(f"来源文件:{metadata['file_name']}")
                if "Title" in metadata: important_meta.append(f"标题:{metadata['Title']}")
                meta_text = " ".join(important_meta)

            # 3. 处理不同类型的内容
            if isinstance(raw_content, dict):
                # 针对金融表格类数据优化：保留键值对映射，并重复关键字段增加权重
                # 例如: {"基金代码": "008124"} -> "基金代码:008124 008124"
                text_parts = []
                for k, v in raw_content.items():
                    if v is not None:
                        text_parts.append(f"{k}:{v}")
                        # 如果是代码类或简称类字段，额外添加一份原文以增强搜索命中
                        if any(key in k for key in ["代码", "简称", "名称", "日期"]):
                            text_parts.append(str(v))
                text = " ".join(text_parts)
            elif isinstance(raw_content, list):
                text = " ".join([str(i) for i in raw_content])
            else:
                text = str(raw_content)

            # 合并元数据和主体内容
            final_text = f"{meta_text} {text}".strip()
            
            if not final_text:
                continue

            # 符号降噪：去除 JSON 特殊符号，压缩空格
            final_text = re.sub(r'["\'{}:,\[\]]', ' ', final_text)
            final_text = re.sub(r'\s+', ' ', final_text)

            # 使用jieba分词
            if jieba:
                seg_list = list(jieba.cut(final_text))
                final_text = " ".join([tok for tok in seg_list if tok])

            # 写入临时文件供 Pyserini 读取
            # 保持 JsonCollection 要求的格式
            with open(os.path.join(json_slice_path, f"{docid}.json"), "w", encoding="utf-8") as fout:
                json.dump({"id": docid, "contents": final_text}, fout, ensure_ascii=False)

    print(f"语料预处理完成，共处理 {cnt} 条数据")

    # 构建索引
    indexing_args = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", json_slice_path,
        "--index", index_path,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "16",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    
    # 如果使用了jieba预分词，使用--pretokenized，否则根据language参数决定
    if jieba:
        print("Using pre-tokenized content (jieba)")
        indexing_args.append("--pretokenized")
    elif language == "zh":
        indexing_args.extend(["--language", "zh"])

    print(f"正在构建 Lucene 索引...")
    subprocess.run(indexing_args, check=True)
    print("BM25 索引构建成功")


if __name__ == "__main__":
    # jsonl_path = "./datasets/database/data_with_embedding_shards/all_data_clean_embedding.jsonl"
    jsonl_path = "datasets/OmniEval-Corpus/all_data_clean.jsonl"
    ner_dict_path = "datasets/NER/finance_ner_dict.txt"

    # 读取文本数据
    if os.path.exists(jsonl_path):
        datasets = load_corpus(jsonl_path)
        print(f"Total documents: {len(datasets)}")
        if len(datasets) > 0:
            print(datasets[0])
    else:
        print(f"Warning: Corpus file not found at {jsonl_path}")

    # BM25 index 生成
    json_slice_path = "./datasets/database/bm25_slices"
    index_path = "./datasets/database/bm25_index"
    
    # 使用优化后的函数，指定自定义词典
    try:
        build_BM25_index(
            jsonl_path=jsonl_path, 
            json_slice_path=json_slice_path, 
            index_path=index_path, 
            language="zh",
            ner_dict_path=ner_dict_path
        )
    except Exception as e:
        print(f"❌ BM25 索引构建失败: {e}")



