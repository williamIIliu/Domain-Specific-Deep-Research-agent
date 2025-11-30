# config文件在verl/verl/trainer/config/sft_trainer.yaml

set -x
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

PROJECT_NAME="BDI"
EXPERIMENT_NAME="Qwen3-8B-SFT-Agentar100K"
SAVE_PATH="../output/generator/${PROJECT_NAME}/${EXPERIMENT_NAME}"
DATA_PATH="../datasets/Agentar-DeepFinance-100K/"
MODEL_PATH="../pretrain_models/generator/Qwen3-8B"

torchrun --standalone --nnodes=1 --nproc_per_node=6 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_PATH}/train.parquet \
    data.val_files=${DATA_PATH}/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length=1500 \
    data.truncation=left \
    data.train_batch_size=12 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=${MODEL_PATH}\
    model.use_liger=False \
    model.enable_gradient_checkpointing=True \
    model.lora_rank=0  \
    model.lora_alpha=64  \
    model.target_modules=all-linear  \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    trainer.max_ckpt_to_keep=5 \
    trainer.total_epochs=2 \
    trainer.seed=42 \
    trainer.test_freq=100 \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.resume_mode=disable \
    trainer.resume_from_path=${SAVE_PATH} \
    optim.lr=1e-4 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    trainer.logger='["swanlab"]' 
