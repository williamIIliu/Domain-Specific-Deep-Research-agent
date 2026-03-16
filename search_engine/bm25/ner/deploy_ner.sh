# 1. 设置调试级别 (DEBUG 会输出最详细的信息，如果信息太多可以改为 INFO)
export VLLM_LOGGING_LEVEL=DEBUG

# 2. 指定只使用编号为 0 和 1 的显卡
export CUDA_VISIBLE_DEVICES=0,1

# 3. 启动服务（注意添加了 --tensor-parallel-size 2）
vllm serve output/ner/Qwen3-4B-final \
  --max-model-len 3596 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen-ner \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 2 \
  --trust-remote-code