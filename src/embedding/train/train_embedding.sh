# ref doc: https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
# uv pip install 
nproc_per_node=1
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model ./pretrain_models/embedding/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type lora \
    --lora_rank 1 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --dataset ./datasets/embedding_finetune/infonce_neg.jsonl \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 2 \
    --save_steps 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3