#!/bin/bash
set -e

export HF_HOME="/scratch/nunes/huggingface_cache/huggingface"


# Paths and ports
# MODEL1_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL2_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL3_NAME="Qwen/Qwen2.5-3B-Instruct"
# MODEL4_NAME="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL5_NAME="Qwen/Qwen3-0.6B"
# MODEL6_NAME="google/gemma-3-270m"


source ~/miniconda3/etc/profile.d/conda.sh
conda activate lotus

cd /home/nunes/lotus-HPC

export VLLM_SERVER_DEV_MODE=1



CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model "meta-llama/Llama-3.1-8B-Instruct" --dtype bfloat16 --tensor-parallel-size 2 --enable-sleep-mode --max-model-len 20000 --gpu-memory-utilization 0.50 2>&1 | tee "/home/nunes/Abacus/palimpzest/abacus-research/saida_VLLM.txt" & PID1=$!
wait $PID1