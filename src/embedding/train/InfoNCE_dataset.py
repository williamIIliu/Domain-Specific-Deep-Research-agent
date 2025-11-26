import json
import time
import torch
import random
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from pymilvus import MilvusClient
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch import Tensor

# -------------------------- 1. 复用你的基础配置（Milvus、模型路径等） --------------------------
# Milvus配置
DB_NAME = "Finance_Corpus"
COLLECTION_NAME = "Finance_RAG_helper"
PARTITION_NAME = "base"
INDEX_FIELD = "embedding"  # Milvus中向量字段名
MILVUS_CONFIG = {
    "uri": "http://localhost:19530",
    "token": "root:Milvus",
    "db_name": DB_NAME
}

# 模型配置（复用你的Embedding模型路径）
MODEL_PATHS = {
    "qwen3": "../../pretrain_weights/embedding/qwen3-0_6b",
    "qwen3_finetune": "../../pretrain_weights/embedding/qwen3-0_6b_finetune",
    "bgem3": "../../pretrain_weights/embedding/bge-m3",
    "gte": "../../pretrain_weights/embedding/gte-large-zh"
}
USE_MODEL = "qwen3"  # 选择要使用的Embedding模型
MAX_SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集配置
ORIGINAL_DATA_PATH = "../../datasets/OmniEval-Corpus/all_data_clean_query.jsonl"  # 你的原始数据（含id、contents、query）
OUTPUT_INFONCE_PATH = "../../datasets/OmniEval-Corpus/embedding_finetune/infonce_neg.jsonl"  # 输出的InfoNCE格式数据
NUM_NEGATIVES = 3  # 每个样本需要的难负样本数量（从Milvus检索结果中选）
RETRIEVE_TOPK = 5  # Milvus检索时取前10个相似结果（从中筛选难负样本）


# -------------------------- 2. 复用你的工具函数（Embedding生成、指令构造） --------------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """提取最后有效token的Embedding（复用你的函数）"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def compute_embeddings(model, tokenizer, text: str) -> List[float]:
    """生成单条文本的Embedding（适配单样本检索场景）"""
    # 文本编码
    batch_dict = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LEN
    ).to(DEVICE)

    # 模型推理（禁用梯度，提升速度）
    model.eval()
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化，适配Milvus的IP/L2距离

    # 转为列表格式（Milvus检索要求）
    return embeddings[0].detach().cpu().numpy().tolist()


def get_detailed_instruct(task_description: str, query: str) -> str:
    """构造带指令的query（复用你的函数，提升检索相关性）"""
    return f'指令: {task_description}\n查询: {query}'


# -------------------------- 3. 核心函数：加载原始数据 + Milvus检索难负样本 + 格式组装 --------------------------
def load_original_data(file_path: str) -> List[Dict]:
    """加载原始数据（筛选含id、contents、query的有效样本）"""
    valid_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                # 必须包含核心字段：id（匹配Milvus中的id）、contents（正样本）、query（锚点）
                if all(key in sample for key in ["id", "contents", "query"]):
                    # 简单清洗内容（避免特殊字符导致JSON解析错误）
                    sample["contents"] = sample["contents"].strip().replace("\n", " ").replace("\r", "")
                    sample["query"] = sample["query"].strip()
                    valid_data.append(sample)
                else:
                    print(f"⚠️ 跳过第{line_num}行：缺少id/contents/query字段")
            except json.JSONDecodeError as e:
                print(f"⚠️ 跳过第{line_num}行：JSON解析错误 - {str(e)[:50]}")

    print(f"✅ 成功加载 {len(valid_data)} 条有效原始样本")
    return valid_data


def retrieve_hard_negatives(
        milvus_client,
        query_embedding: List[float],
        current_sample_id: str,
        topk: int = 10,
        num_samples: int = 1,
        num_filter_samples: int = 3,
) -> List[str]:
    """
    从 Milvus 检索难负样本（相似但非自身的文档内容），过滤前若干条结果，只采样指定数量。

    :param milvus_client: Milvus客户端实例
    :param query_embedding: 当前样本 query 的 Embedding
    :param current_sample_id: 当前样本的 id（用于排除自身）
    :param topk: 检索的总返回数量（建议 > num_filter_samples + num_samples）
    :param num_samples: 要保留的难负样本数量（最终输出数量）
    :param num_filter_samples: 跳过前多少条最相似样本（避免自身或过近样本）
    :return: 难负样本 contents 列表
    """
    try:
        # 调用 Milvus 检索
        search_result = milvus_client.search(
            collection_name=COLLECTION_NAME,
            partition_names=[PARTITION_NAME],
            anns_field=INDEX_FIELD,
            data=[query_embedding],
            limit=topk,
            output_fields=["id", "text_chunk"],
        )

        hard_candidates = []
        for hit in search_result[0]:
            retrieved_id = str(hit["entity"]["id"])
            retrieved_text = hit["entity"]["text_chunk"].strip()

            # 过滤自身与空文本
            if retrieved_id != current_sample_id and retrieved_text:
                hard_candidates.append(retrieved_text)

        # 如果可用候选不足，则返回空（不填充）
        if len(hard_candidates) <= num_filter_samples:
            print(f"⚠️ 有效候选不足（共 {len(hard_candidates)} 条，过滤 {num_filter_samples} 条） -> 返回空")
            return []

        # 过滤掉最相似的前 num_filter_samples 条
        filtered_candidates = hard_candidates[num_filter_samples:]

        # 如果剩余数量不足所需采样数，则直接全部返回（不重复）
        if len(filtered_candidates) < num_samples:
            print(f"⚠️ 难负样本数量不足，仅返回 {len(filtered_candidates)} 条")
            return filtered_candidates

        # 随机选取指定数量
        selected = random.sample(filtered_candidates, num_samples)
        return selected

    except Exception as e:
        print(f"❌ 检索难负样本失败：{str(e)[:120]}")
        return []

def sample_easy_negatives(
    original_data: List[Dict[str, Any]],
    current_sample_id: str,
    current_content: Any,
    num_negatives: int
) -> List[Any]:
    """
    从数据集中随机采样简单负样本（Simple Negatives）
    要求：
      1. 不能是当前样本自身；
      2. contents 的类型必须与当前样本相同；
      3. 负样本之间不重复。

    :param original_data: 全部样本数据（列表）
    :param current_sample_id: 当前样本ID（用于排除自身）
    :param current_content: 当前样本的 contents 值（用于匹配类型）
    :param num_negatives: 要采样的负样本数量
    :return: 随机负样本的 contents 列表
    """
    # 确定当前样本的内容类型
    target_type = type(current_content)

    # 构造候选集：排除自身 + 类型匹配
    candidates = [
        sample["contents"] for sample in original_data
        if sample["id"] != current_sample_id and isinstance(sample["contents"], target_type)
    ]

    # 去重（防止重复内容）
    candidates = list(set(map(str, candidates)))  # 先用 str 去重，再转回
    candidates = [json.loads(c) if c.startswith("{") or c.startswith("[") else c for c in candidates]

    # 候选不足时直接全部使用
    if len(candidates) <= num_negatives:
        return random.sample(candidates, len(candidates))

    # 否则随机采样 num_negatives 个
    return random.sample(candidates, num_negatives)

def build_infonce_dataset(
        original_data: List[Dict],
        milvus_client: MilvusClient,
        model,
        tokenizer
) -> None:
    """
    构建InfoNCE格式数据集（锚点+正样本+Milvus检索的难负样本）
    输出JSONL文件，每行一个InfoNCE样本
    """
    # 打开输出文件（追加模式，支持中断后继续生成）
    with open(OUTPUT_INFONCE_PATH, "a", encoding="utf-8") as f:
        # 进度条显示
        for sample in tqdm(original_data, desc="🔨 构建InfoNCE数据集"):
            try:
                # 1. 提取当前样本的核心信息
                current_id = str(sample["id"])
                anchor_query = sample["query"]  # 锚点：原始样本的query
                positive_content = sample["contents"]  # 正样本：原始样本的contents

                # 2. 生成带指令的query Embedding（提升检索相关性）
                task_desc = "检索与当前查询相似但不相关的金融文档，用于对比学习训练"
                instructed_query = get_detailed_instruct(task_desc, anchor_query)
                query_embedding = compute_embeddings(model, tokenizer, instructed_query)

                # 3. 用Milvus检索难负样本
                hard_negatives = retrieve_hard_negatives(
                    milvus_client=milvus_client,
                    query_embedding=query_embedding,
                    current_sample_id=current_id,
                    topk=5,
                    num_samples=2,  # 只取一个
                    num_filter_samples=3,  # 跳过前3个
                )

                # 4. 随机简单负样本（数量可自定义，比如 2 个）
                simple_negatives = sample_easy_negatives(
                    original_data=original_data,
                    current_sample_id=current_id,
                    current_content=positive_content,
                    num_negatives=2
                )

                # 4. 组装InfoNCE格式（严格符合之前定义的结构）
                # 合并
                if not hard_negatives:
                    all_negatives = list(dict.fromkeys(hard_negatives + simple_negatives))
                else:
                    all_negatives = simple_negatives
                infonce_sample = {
                    "messages": [{"role": "user", "content": anchor_query}],  # 锚点（query）
                    "positive_messages": [  # 正样本（当前样本的contents）
                        [{"role": "user", "content": positive_content}]
                    ],
                    "negative_messages": [  # 难负样本（Milvus检索到的相似文档）
                        [{"role": "user", "content": neg_content}] for neg_content in all_negatives
                    ]
                }

                # 5. 写入输出文件（JSONL格式）
                json.dump(infonce_sample, f, ensure_ascii=False)
                f.write("\n")

            except Exception as e:
                # 单个样本处理失败，记录日志并跳过（不中断整体流程）
                error_msg = f"样本{current_id[:8]}...处理失败：{str(e)[:100]}"
                print(f"⚠️ {error_msg}")
                # 可选：将失败样本记录到日志文件
                with open("dataset_build_error.log", "a", encoding="utf-8") as err_f:
                    err_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {error_msg}\n")

    print(f"\n🎉 InfoNCE数据集构建完成！输出路径：{OUTPUT_INFONCE_PATH}")
    # 统计成功生成的样本数
    with open(OUTPUT_INFONCE_PATH, "r", encoding="utf-8") as f:
        success_count = len([line for line in f if line.strip()])
    print(f"📊 成功生成 {success_count} 条InfoNCE样本（每条含{NUM_NEGATIVES}个难负样本）")


# -------------------------- 4. 主函数：串联所有流程 --------------------------
def main():
    try:
        print("=" * 80)
        print("🚀 开始基于Milvus的InfoNCE数据集构建流程")
        print("=" * 80)

        # 步骤1：初始化Milvus客户端（复用你的配置）
        print("\n1. 初始化Milvus客户端")
        milvus_client = MilvusClient(**MILVUS_CONFIG)
        # 检查集合是否存在且已加载
        if not milvus_client.has_collection(collection_name=COLLECTION_NAME):
            raise ValueError(f"Milvus集合 {COLLECTION_NAME} 不存在，请先创建并插入数据")
        # 加载集合（若未加载）
        # if not milvus_client.get_collection_load_state(collection_name=COLLECTION_NAME):
        milvus_client.load_collection(collection_name=COLLECTION_NAME)
        print("✅ Milvus客户端初始化完成（集合已加载）")
        # 重新确定索引
        # 1. IVF-FLAT
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 1024},  # 分成多少簇, 越多划分越细，检索速度更慢； 参数从32-4096， 我们设置128
        )

        # 步骤2：加载Embedding模型和Tokenizer（复用你的模型路径）
        print(f"\n2. 加载Embedding模型：{USE_MODEL}")
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATHS[USE_MODEL],
            local_files_only=True,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATHS[USE_MODEL],
            max_length=MAX_SEQ_LEN
        )
        print("✅ Embedding模型和Tokenizer加载完成")

        # 步骤3：加载原始数据（含id、contents、query）
        print(f"\n3. 加载原始数据：{ORIGINAL_DATA_PATH}")
        original_data = load_original_data(ORIGINAL_DATA_PATH)
        if not original_data:
            raise ValueError("❌ 无有效原始数据，流程终止")

        # 步骤4：构建InfoNCE数据集（核心步骤）
        print(f"\n4. 开始构建InfoNCE数据集（Milvus检索Top{RETRIEVE_TOPK}，选{NUM_NEGATIVES}个难负样本）")
        build_infonce_dataset(original_data, milvus_client, model, tokenizer)

        print("\n" + "=" * 80)
        print("🎉 所有流程完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 流程中断：{str(e)}")
        traceback.print_exc()
        print("=" * 80)


if __name__ == "__main__":
    import traceback  # 延迟导入，仅在主函数异常时使用

    main()