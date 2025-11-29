# config文件在verl/verl/trainer/config/sft_trainer.yaml

set -x
export CUDA_VISIBLE_DEVICES=0,1
trainer.project_name="BDI"
trainer.experiment_name="Qwen3-8B-SFT-Agentar100K"
save_path = "output/generator/${trainer.project_name}/${trainer.experiment_name}"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=../datasets/Agentar-DeepFinance-100K/train.parquet \
    data.val_files=../datasets/Agentar-DeepFinance-100K/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length=1508 \
    optim.lr=1e-4 \
    data.micro_batch_size=4 \
    model.partial_pretrain=../pretrain_models/generator/Qwen3-8B \
    model.use_liger=True \
    model.lora_rank: 32  \
    model.lora_alpha: 64  \
    model.target_modules: all-linear  \
    model.fsdp_config.model_dtype: bf16 \
    model.fsdp_config.cpu_offload: True \
    model.fsdp_config.offload_params: True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=${trainer.project_name} \
    trainer.experiment_name=${trainer.experiment_name} \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.logger='["swanlab"]' 
