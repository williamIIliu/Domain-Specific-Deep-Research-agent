set -x

torchrun --standalone --nnodes=2 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=../datasets/Agentar-DeepFinance-100K/train.parquet \
    data.val_files=../datasets/Agentar-DeepFinance-100K/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=../pretrain_models/generator/Qwen3-8B \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=BDI \
    trainer.experiment_name=Qwen3-8B-SFT-Agentar100K \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.logger='["swanlab"]' \
    save_path=../output/Qwen3-8B-SFT-Agentar100K 
