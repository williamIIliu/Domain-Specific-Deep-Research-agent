# ./pretrain_models/reward_model/FinR1
CUDA_VISIBLE_DEVICES=2  
python -m sglang.launch_server \
--model-path ./pretrain_models/generator/Qwen3-0.6B \
--trust-remote-code \ 
--served-model-name reward_model \
--tensor-parallel-size 1 \
--mem-fraction-static 0.90 \
--api-key sk-123456 \
--host 0.0.0.0 --port 8000 \
--tool-call-parser qwen25 \
--reasoning-parser qwen3 \
--context-length 131072 