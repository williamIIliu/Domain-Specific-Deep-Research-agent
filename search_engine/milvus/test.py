import os
import json
import time
from typing import Any, Dict, List
from rank_bm25 import BM25Okapi

import numpy as np
import torch
from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
    Function,
    FunctionType,
)

# ==========================
# 1️⃣ 连接 Milvus
# ==========================
DB_NAME = "Finance_Corpus"
COLLECTION_NAME = "Finance_RAG_helper_hybrid"
PARTITION_NAME = "base"
DATA_PATH = "../../datasets/OmniEval-Corpus/demo/test_embedding.jsonl"

root_client = MilvusClient(
    uri="http://localhost:19530",
    user="root",
    password="Milvus",
)

# 检查数据库是否存在
db_list = root_client.list_databases()
if DB_NAME in db_list:
    print(f"🔁 检查数据库 {DB_NAME} ...")
    # ⚠️ 先连接到该数据库，删除所有 collection
    temp_client = MilvusClient(
        uri="http://localhost:19530",
        user="root",
        password="Milvus",
        db_name=DB_NAME,
    )
    collections = temp_client.list_collections()
    for c in collections:
        print(f"🗑️ 删除旧 collection: {c}")
        temp_client.drop_collection(c)

    # 然后再安全删除数据库
    print(f"🗑️ 删除旧数据库: {DB_NAME}")
    root_client.drop_database(DB_NAME)

# 重新创建数据库
print(f"✅ 创建新数据库: {DB_NAME}")
root_client.create_database(DB_NAME)

# 连接到新数据库
client = MilvusClient(
    uri="http://localhost:19530",
    user="root",
    password="Milvus",
    db_name=DB_NAME,
)
# ==========================
# 2️⃣ 定义 Schema
# ==========================
analyzer_params = {"type": "chinese"}  # 中文分词器

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
    FieldSchema(
        name="text_chunk",
        dtype=DataType.VARCHAR,
        max_length=1024,
        enable_analyzer=True,
        analyzer_params=analyzer_params,
        enable_match=True,
    ),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_bm25", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="publish_time", dtype=DataType.INT64),
    FieldSchema(name="metadata", dtype=DataType.JSON, enable_dynamic=True),
]

schema = CollectionSchema(fields,
                          enable_dynamic_field=True,
                          description="Finance RAG helper collection")

bm25_function = Function(
    name="bm25_fn",
    input_field_names=["text_chunk"],
    output_field_names="sparse_bm25",
    function_type=FunctionType.BM25,
)

schema.add_function(bm25_function)

# ==========================
# 3️⃣ 定义索引
# ==========================
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    index_name="embedding_index",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 1024},
)

index_params.add_index(
    field_name="sparse_bm25",
    index_name="sparse_bm25_index",
    index_type="sparse_inverted_index",
    metric_type="BM25",
    params={"inverted_index_algo": "DAAT_MAXSCORE"}, # Algorithm used for building and querying the index
)


# ==========================
# 4️⃣ 创建 Collection 和分区
# ==========================
collections = client.list_collections()
if COLLECTION_NAME in collections:
    print(f"🗑️ 删除旧的 collection: {COLLECTION_NAME}")
    client.drop_collection(COLLECTION_NAME)

print(f"📦 创建新的 collection: {COLLECTION_NAME}")
collection = client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
)

# 创建分区
# partitions = client.list_partitions(COLLECTION_NAME)
# if PARTITION_NAME not in partitions:
#     client.create_partition(COLLECTION_NAME, PARTITION_NAME)
#     print(f"📂 创建分区: {PARTITION_NAME}")

# ==========================
# 5️⃣ 插入数据
# ==========================
dataset = []
print(f"📖 正在加载数据: {DATA_PATH}")

import jieba
# 读取数据
records = []
texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        records.append(data)
        tokens = list(jieba.cut(data["text_chunk"]))
        texts.append(tokens)

# Step 1: 对文本做中文分词
tokenized_texts = [list(jieba.cut(d["text_chunk"])) for d in dataset]

# Step 2: 构建 BM25 模型
bm25 = BM25Okapi(tokenized_texts)

# Step 3: 构建全词表
vocab = list(set(token for doc in tokenized_texts for token in doc))
vocab_index = {word: i for i, word in enumerate(vocab)}

# Step 4: 为每条记录生成 sparse BM25 vector
for i, record in enumerate(dataset):
    tokens = tokenized_texts[i]
    scores = bm25.get_scores(tokens)
    sparse_vector = [0.0] * len(vocab)
    for t in tokens:
        idx = vocab_index[t]
        sparse_vector[idx] = float(scores[idx])
    record["sparse_bm25"] = sparse_vector

batch_size = 500
n = len(dataset)
total_time = 0.0
print(f"📦 总计待插入数据: {n} 条")

for i in range(0, n, batch_size):
    batch = dataset[i: i + batch_size]
    start = time.time()
    client.insert(
        collection_name=COLLECTION_NAME,
        data=batch,
        partition_name=PARTITION_NAME,
    )
    elapsed = time.time() - start
    total_time += elapsed
    print(f"✅ 已插入 {min(i + batch_size, n)} / {n} 条，用时 {elapsed:.2f} 秒")

print(f"🏁 总耗时 {total_time:.2f} 秒，平均 {(total_time / n) * 1000:.2f} ms/条")

# ==========================
# ✅ 验证 Schema
# ==========================
info = client.describe_collection(COLLECTION_NAME)
for field in info["fields"]:
    print(f"字段 {field['name']} nullable: {field.get('nullable', False)}")

print("🎉 Collection 创建与数据插入全部完成！")
