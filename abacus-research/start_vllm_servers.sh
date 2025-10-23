#!/bin/bash
set -e

export HF_HOME="/scratch/global/huggingface_cache/huggingface"



# Paths and ports
MODEL1_NAME="Qwen/Qwen3-4B"
MODEL2_NAME="Qwen/Qwen3-4B"
MODEL3_NAME="Qwen/Qwen3-4B"
# "meta-llama/Llama-3.1-8B-Instruct"
# "Qwen/Qwen2.5-1.5B-Instruct"
# "Qwen/Qwen2.5-0.5B-Instruct"
# "Qwen/Qwen3-0.6B"
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

EMBEDDING_PORT=8001
MODEL1_PORT=8105
MODEL2_PORT=8106
MODEL3_PORT=8107

LOG_DIR="/home/nunes/Abacus/palimpzest/abacus-research/var/logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lotus


wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}


echo "[INFO] Starting vLLM embedding server"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model nomic-ai/nomic-embed-text-v1 --task embed --port $EMBEDDING_PORT --trust-remote-code --max-model-len 8K &
PIDEmbedding=$!
wait_for_ready $EMBEDDING_PORT


echo "[INFO] Starting vLLM inference servers"
#checar se isso é seguro
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model "$MODEL1_NAME" --port $MODEL1_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.7  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" &
PID1=$!
wait_for_ready $MODEL1_PORT


CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model "$MODEL2_NAME" --port $MODEL2_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.7 2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
PID2=$!
wait_for_ready $MODEL2_PORT

# CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --tensor-parallel-size 2 --model "$MODEL3_NAME" --port $MODEL3_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.3  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL3_PORT.txt" &
# PID3=$!
# wait_for_ready $MODEL3_PORT




wait $PID1 $PID2 # $PID3 $PIDEmbedding