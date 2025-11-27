import os
import json
import numpy as np
import faiss
from tqdm import tqdm


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


def build_faiss_index(jsonl_path, save_path, batch_size=1000):
    """
    从 JSONL 数据构建 FAISS 向量索引并保存
    """
    print(f"📂 正在从 {jsonl_path} 加载向量并构建索引...")

    index = None
    all_ids = []

    for embeddings, ids in tqdm(load_embeddings(jsonl_path, batch_size=batch_size)):
        if index is None:
            dim = embeddings.shape[1]
            # 这里使用简单的 L2 距离索引，也可改用 IndexFlatIP（内积相似度）
            index = faiss.IndexFlatL2(dim)

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

    print("🎉 所有工作完成！")


if __name__ == "__main__":
    jsonl_path = "./datasets/OmniEval-Corpus/all_data_clean_embedding.jsonl"   # 输入 JSONL 路径
    save_path = "./datasets/DB/Qwen3"      # 输出保存目录
    batch_size = 1024                # 批次大小，可根据内存调整

    os.makedirs(save_path, exist_ok=True)
    build_faiss_index(jsonl_path, save_path, batch_size)
