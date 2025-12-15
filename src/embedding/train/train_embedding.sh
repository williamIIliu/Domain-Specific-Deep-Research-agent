# ref doc: https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
# uv pip install ms-swift
# INFONCE_MASK_FAKE_NEGATIVE=true:过滤掉假负样本，也就是负样本的相似度超过正样本的
# https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html

export INFONCE_MASK_FAKE_NEGATIVE=true 
export CUDA_VISIBLE_DEVICES=0,1

PROJECT_NAME="BDI"
EXPERIMENT_NAME="Qwen3-Embedding-0.6B-Finetune"
MODEL_PATH="./pretrain_models/embedding/Qwen3-Embedding-0.6B"
SAVE_PATH="./output/embedding/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# 使用SWIFT_NPROC_PER_NODE启动多卡训练
NPROC_PER_NODE=2 swift sft \
    --model ${MODEL_PATH} \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --lora_rank 0 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --warmup_ratio 0.05 \
    --dataset ./datasets/embedding/infonce_qa_dataset.jsonl \
    --max_length 1310 \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir ${SAVE_PATH} \
    --eval_steps 100 \
    --num_train_epochs 2 \
    --save_steps 100 \
     --output_dir  ./pretrain_models/embedding/Qwen3-0.6B_finetune \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3 \
    --logging_steps 10 \
    --report_to swanlab \
    --swanlab_project ${PROJECT_NAME} \
    --swanlab_exp_name ${EXPERIMENT_NAME} \
    --logging_dir ./logs/embedding 