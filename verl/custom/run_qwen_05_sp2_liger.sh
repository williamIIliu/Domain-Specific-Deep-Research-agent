# config文件在verl/verl/trainer/config/sft_trainer.yaml

set -x
export CUDA_VISIBLE_DEVICES=0,1

PROJECT_NAME="BDI"
EXPERIMENT_NAME="Qwen3-8B-SFT-Agentar100K"
SAVE_PATH="../output/generator/${PROJECT_NAME}/${EXPERIMENT_NAME}"
DATA_PATH="../datasets/Agentar-DeepFinance-100K/"
MODEL_PATH="../pretrain_models/generator/Qwen3-8B"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_PATH}/train.parquet \
    data.val_files=${DATA_PATH}/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length=1508 \
    optim.lr=1e-4 \
    data.micro_batch_size=4 \
    model.partial_pretrain=${MODEL_PATH}\
    model.use_liger=True \
    model.lora_rank: 32  \
    model.lora_alpha: 64  \
    model.target_modules: all-linear  \
    model.fsdp_config.model_dtype: bf16 \
    model.fsdp_config.cpu_offload: True \
    model.fsdp_config.offload_params: True \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.logger='["swanlab"]' 
