#!/usr/bin/env bash
export EVALSCOPE_PERF_SKIP_DB=1
export CUDA_VISIBLE_DEVICES=0, 1

evalscope perf \
  --parallel 1 10 50 100 200 \
  --number 10 20 100 200 400 \
  --model Fin-Search \
  --url http://127.0.0.1:11434/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --max-tokens 1024 \
  --prefix-length 0 \
  --max-prompt-length 1024 \
  --tokenizer-path ./pretrain_models/generator/Qwen3-8B/ \
  --extra-args '{"ignore_eos": true}' \
  --visualizer swanlab \
  --outputs-dir output/evalscope/Qwen3-8B-Q4_K_M/