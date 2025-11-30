# ./pretrain_models/reward_model/FinR1
CUDA_VISIBLE_DEVICES=2  
python -m sglang.launch_server \
--model-path ./pretrain_models/generator/Qwen3-0.6B \
--trust-remote-code \
--dtype bfloat16 \
--served-model-name reward_model \
--max-total-tokens 1024 \
--tensor-parallel-size 1 \
--mem-fraction-static 0.20 \
--api-key sk-123456 \
--host 0.0.0.0 --port 8060 \
--max-running-requests 4 \
--context-length 1024 