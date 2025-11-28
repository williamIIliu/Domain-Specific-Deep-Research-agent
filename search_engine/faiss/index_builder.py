import os
import json
import numpy as np
import faiss
from tqdm import tqdm
import subprocess

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

                # 处理embedding字段
                item["embedding"] = data.get("embedding", "")
                corpus.append(item)

            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败 - {str(e)}")
            except Exception as e:
                print(f"警告：第{line_num}行处理失败 - {str(e)}")

    return corpus

# ------------------------- FAISS Index -------------------------
def build_faiss_index(jsonl_path, save_path, batch_size=1000):
    """
    从 JSONL 数据构建 FAISS 向量索引并保存
    """

    def load_embeddings(jsonl_path, batch_size=1000):
        """
        分批加载 JSONL 文件中的 embedding 向量和对应 id
        """
        embeddings_batch = []
        ids_batch = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                emb = obj.get("embedding", None)
                if emb is not None:
                    embeddings_batch.append(np.array(emb, dtype=np.float32))
                    ids_batch.append(obj.get("id"))

                # 批量返回
                if len(embeddings_batch) >= batch_size:
                    yield np.stack(embeddings_batch, axis=0), ids_batch
                    embeddings_batch, ids_batch = [], []

        # 返回剩余部分
        if embeddings_batch:
            yield np.stack(embeddings_batch, axis=0), ids_batch

    print(f"📂 正在从 {jsonl_path} 加载向量并构建索引...")

    index = None
    all_ids = []

    for embeddings, ids in tqdm(load_embeddings(jsonl_path, batch_size=batch_size)):
        if index is None:
            dim = embeddings.shape[1]
            # 这里使用简单的 L2 距离索引，也可改用 IndexFlatIP（内积相似度）
            index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        all_ids.extend(ids)

    if index is None:
        raise ValueError("❌ 没有加载到任何 embedding 数据。")

    print(f"✅ 索引构建完成，共 {len(all_ids)} 条向量。")

    # 保存 faiss 索引
    faiss_index_path = os.path.join(save_path, "faiss_index.bin")
    faiss.write_index(index, faiss_index_path)
    print(f"💾 FAISS 索引已保存到: {faiss_index_path}")

    # 保存 id 对应表
    id_map_path = os.path.join(save_path, "id_map.json")
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)
    print(f"💾 ID 映射表已保存到: {id_map_path}")
    print("FAISS 保存工作完成！")


# ------------------------- BM25 Index -------------------------
def build_BM25_index(jsonl_path, json_slice_path, index_path):
    # 创建目录
    os.makedirs(json_slice_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)

    # 拆分JSONL为单个文件
    cnt = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="拆分文件"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cnt += 1
            docid = obj["id"]
            # 提取文本内容，确保数字字段也被索引
            raw_content = obj.get("contents") or obj.get("text_chunk") or obj.get("context") or ""
            
            if isinstance(raw_content, dict):
                # 将字典的所有键值对转为 "键:值" 格式，数字也转为字符串
                # 例如: {"股票代码":"002851","行业名称":"电力设备"} 
                # 转为: "股票代码:002851 行业名称:电力设备"
                text = " ".join([f"{k}:{v}" for k, v in raw_content.items() if v is not None])
            elif isinstance(raw_content, list):
                text = json.dumps(raw_content, ensure_ascii=False)
            else:
                text = str(raw_content)
            if not text.strip():
                continue
            # 写入文件
            with open(f"{json_slice_path}/{docid}.json", "w", encoding="utf-8") as fout:
                json.dump({"id": docid, "contents": text}, fout, ensure_ascii=False)

    print(f"拆分完成，处理{cnt}条数据")

    # 构建索引
    subprocess.run([
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", json_slice_path,
        "--index", index_path,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ], check=True)
    print("BM25索引构建完成")


if __name__ == "__main__":
    jsonl_path = "./datasets/database/data_with_embedding_shards/all_data_clean_embedding.jsonl"

    # 读取文本数据
    datasets = load_corpus(jsonl_path)
    print(len(datasets))
    print(datasets[0])

    # FAISS index 生成
    save_path = "./datasets/database/faiss_qwen"  # 输出保存目录
    batch_size = 1024  # 批次大小，可根据内存调整
    os.makedirs(save_path, exist_ok=True)
    build_faiss_index(jsonl_path, save_path, batch_size)

    # BM25 index 生成
    json_slice_path = "./datasets/database/bm25_tokenize"
    index_path = "./datasets/database/bm25"
    build_BM25_index(jsonl_path, json_slice_path, index_path)



