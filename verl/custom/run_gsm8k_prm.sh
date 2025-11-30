# make sure your current working directory is the root of the project
# export RAY_DEDUP_LOGS=0
# export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=0,1

set -x
ray start --head --num-gpus=2
ulimit -n 65535

TRAIN_DATA_PATH="../datasets/gsm_prm_reward_test/train.parquet"
VAL_DATA_PATH="../datasets/gsm_prm_reward_test/test.parquet"

PROJECT_NAME="BDI"
EXPERIMENT_NAME="Qwen3-8B-gsm8k-GRPO-PRM"

MODEL_PATH="../pretrain_models/generator/Qwen3-0.6B"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.15 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=5 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=False \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.default_local_dir="../output/${EXPERIMENT_NAME}" \
    trainer.resume_mode=disable \
    trainer.logger='["swanlab", "console"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME $@
