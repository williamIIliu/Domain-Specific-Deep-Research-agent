# ref doc: https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
# uv pip install ms-swift
# INFONCE_MASK_FAKE_NEGATIVE=true:过滤掉假负样本，也就是负样本的相似度超过正样本的

export INFONCE_MASK_FAKE_NEGATIVE=true 
export CUDA_VISIBLE_DEVICES=0,1 

swift sft \
    --model ./pretrain_models/embedding/Qwen3-Embedding-0.6B \
    --nproc_per_node 2 \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --dataset ./datasets/embedding/infonce_neg.jsonl \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 2 \
    --save_steps 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3 \
    --logging_steps 10 \
    --report_to wandb \
    --wandb_project BDI \
    --run_name qwen3_emb_0.6b_lora_infonce \
    --logging_dir ./logs