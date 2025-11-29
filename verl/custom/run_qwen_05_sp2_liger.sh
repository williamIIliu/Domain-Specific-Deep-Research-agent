set -x
export CUDA_VISIBLE_DEVICES=0,1

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=../datasets/Agentar-DeepFinance-100K/train.parquet \
    data.val_files=../datasets/Agentar-DeepFinance-100K/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    optim.lr=1e-4 \
    data.micro_batch_size=4 \
    model.partial_pretrain=../pretrain_models/generator/Qwen3-8B \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=BDI \
    trainer.experiment_name=Qwen3-8B-SFT-Agentar100K \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.logger='["swanlab"]' 
