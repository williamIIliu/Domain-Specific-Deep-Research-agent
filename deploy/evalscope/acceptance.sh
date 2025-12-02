#!/usr/bin/env bash

evalscope perf \
  --parallel 1 10 50 100 200 \
  --number 10 20 100 200 400 \
  --model Fin-Search \
  --url http://127.0.0.1:11434/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 50 \
  --max-prompt-length 1024 \
  --tokenizer-path ./pretrain_models/generator/Qwen3-8B/ \
  --debug \
  --visualizer swanlab \
  --outputs-dir output/evalscope/Qwen3-8B-Q4_K_M/ \
  --skip-db \ 