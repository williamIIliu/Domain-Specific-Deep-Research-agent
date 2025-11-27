import json
import os
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast  # 混合精度推理
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging


# ---------------------- 1. 基础配置：日志+参数解析 ----------------------
def setup_logging(local_rank):
    """配置日志"""
    logger = logging.getLogger(__name__)
    log_format = f"%(asctime)s - [GPU-{local_rank}] %(message)s"
    formatter = logging.Formatter(log_format)
    logger.setLevel(logging.INFO if local_rank == 0 else logging.ERROR)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args():
    """解析参数（移除--local_rank，改为从环境变量获取）"""
    parser = argparse.ArgumentParser(description="7GPU分布式Embedding推理")
    # 移除--local_rank参数，不再从命令行接收
    parser.add_argument("--jsonl_path", type=str,
                        default="../../datasets/OmniEval-Corpus/all_data_clean.jsonl",
                        help="输入JSONL数据路径")
    parser.add_argument("--model_path", type=str,
                        default="../../pretrain_weights/embedding/qwen3-0_6b",
                        help="Embedding模型路径")
    parser.add_argument("--save_dir", type=str,
                        default="../../datasets/OmniEval-Corpus/embedding_shards",
                        help="分片结果保存目录")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="单GPU批次大小")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="文本最大长度")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader工作线程数")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="启用FP16混合精度")
    return parser.parse_args()


# ---------------------- 2. 数据集优化：适配分布式 ----------------------
class JsonlDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    mapped = self.data_map(record)
                    self.samples.append(mapped)
                except json.JSONDecodeError as e:
                    logging.warning(f"跳过无效行 {i + 1}：{str(e)}")
                    continue

    def data_map(self, record):
        rid = record.get("id", "")
        text_chunk = record.get("contents", "")
        if isinstance(text_chunk, dict):
            text_chunk = json.dumps(text_chunk, ensure_ascii=False, separators=(',', ':'))

        metadata = record.get("metadata", {})
        last_modified = metadata.get("last_modified_date", "")
        publish_time = int(last_modified.replace("-", "")) if (last_modified and "-" in last_modified) else 0
        source = metadata.get("file_name") or metadata.get("source_file") or ""

        return {
            "id": rid,
            "text_chunk": text_chunk,
            "publish_time": publish_time,
            "source": source,
            "metadata": {"score": 0.0}
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        encoding = self.tokenizer(
            rec["text_chunk"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True
        )
        return rec, {k: v.squeeze(0) for k, v in encoding.items()}

    @staticmethod
    def collate_fn(batch):
        recs = [item[0] for item in batch]
        encodings = {}
        for k in batch[0][1].keys():
            encodings[k] = torch.stack([item[1][k] for item in batch], dim=0)
        return recs, encodings


# ---------------------- 3. Embedding计算优化 ----------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    device = last_hidden_states.device
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1].to(device)
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=device),
            sequence_lengths
        ].to(device)


def compute_embeddings(model, batch_dict, device, fp16=True):
    with autocast(enabled=fp16):
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    return embeddings


# ---------------------- 4. 分布式推理核心逻辑 ----------------------
def distributed_embed_and_write(args, local_rank, logger):
    # 初始化分布式环境（使用环境变量指定的local_rank）
    dist.init_process_group(backend="nccl")

    # 设置设备（直接使用传入的local_rank）
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger.info(f"分布式环境初始化完成：GPU-{local_rank}，总进程数={dist.get_world_size()}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    shard_save_path = os.path.join(args.save_dir, f"embedding_shard_{local_rank}.jsonl")
    logger.info(f"GPU-{local_rank} 分片文件路径：{shard_save_path}")

    # 加载模型和Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        local_files_only=True,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.float16
    )
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model.eval()
    logger.info(f"GPU-{local_rank} 模型加载完成（FP16={args.fp16}）")

    # 构建数据集和数据加载器
    dataset = JsonlDataset(
        jsonl_path=args.jsonl_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=JsonlDataset.collate_fn,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False
    )
    logger.info(f"GPU-{local_rank} 数据加载完成：总样本数={len(dataset)}，批次数量={len(dataloader)}")

    # 推理并写入结果
    with torch.no_grad(), open(shard_save_path, "w", encoding="utf-8") as f:
        pbar = tqdm(dataloader, desc=f"GPU-{local_rank} 推理中", disable=(local_rank != 0))
        for batch_idx, (recs, encodings) in enumerate(pbar):
            batch_dict = {k: v.to(device) for k, v in encodings.items()}
            embeddings = compute_embeddings(model, batch_dict, device, args.fp16)

            for rec, emb in zip(recs, embeddings.cpu().numpy()):
                rec["embedding"] = emb.tolist()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            if local_rank == 0:
                total_processed = (batch_idx + 1) * args.batch_size * dist.get_world_size()
                pbar.set_postfix({"已处理样本": total_processed})

    # 等待所有进程完成后合并分片
    dist.barrier()
    logger.info(f"GPU-{local_rank} 推理完成！分片文件：{shard_save_path}")

    if local_rank == 0:
        merge_shards(args.save_dir, os.path.join(args.save_dir, "all_data_clean_embedding.jsonl"), logger)
        logger.info("所有分片合并完成！")

    dist.destroy_process_group()


def merge_shards(shard_dir, merge_save_path, logger):
    shard_files = [f for f in os.listdir(shard_dir)
                   if f.startswith("embedding_shard_") and f.endswith(".jsonl")]
    shard_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    with open(merge_save_path, "w", encoding="utf-8") as merge_f:
        for shard_file in shard_files:
            shard_path = os.path.join(shard_dir, shard_file)
            logger.info(f"合并分片：{shard_path}")
            with open(shard_path, "r", encoding="utf-8") as shard_f:
                for line in shard_f:
                    merge_f.write(line)


# ---------------------- 5. 主函数入口（手动指定local_rank） ----------------------
if __name__ == "__main__":
    args = parse_args()
    # 从环境变量获取local_rank（用户手动指定）
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        # 允许用户在代码中固定local_rank（单机多卡场景下手动指定0~6）
        # 例如：local_rank = 0  # 仅测试单卡时使用，分布式时注释掉
        raise ValueError("请通过环境变量设置LOCAL_RANK（例如：export LOCAL_RANK=0）")

    logger = setup_logging(local_rank)

    try:
        distributed_embed_and_write(args, local_rank, logger)
    except Exception as e:
        logger.error(f"GPU-{local_rank} 执行失败：{str(e)}", exc_info=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
