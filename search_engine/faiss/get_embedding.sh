python -m torch.distributed.run --nproc_per_node=7 \
  search_engine/faiss/get_embedding.py \
  --jsonl_path="./datasets/OmniEval-Corpus/all_data_clean_new.jsonl" \
  --model_path="./pretrain_models/embedding/Qwen3-Embedding-0.6B" \
  --save_dir="./datasets/database/data_with_embedding_shards" \
  --num_workers=2 \
  --batch_size=128 \
  --max_length=1024 \
  --fp16