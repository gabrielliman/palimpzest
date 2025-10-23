#!/bin/bash
set -e

export HF_HOME="/scratch/global/huggingface_cache/huggingface"



# Paths and ports
MODEL1_NAME="Qwen/Qwen3-4B"
MODEL2_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL3_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL4_NAME="Qwen/Qwen3-4B"
MODEL5_NAME="Qwen/Qwen3-4B"
# MODEL5_NAME="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL5_NAME="Qwen/Qwen3-0.6B"
MODEL6_NAME="Qwen/Qwen3-4B"
#"google/gemma-3-270m"

# MODEL3_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

EMBEDDING_PORT=8001
MODEL1_PORT=8002
MODEL2_PORT=8003
MODEL3_PORT=8004
MODEL4_PORT=8105
MODEL5_PORT=8106
MODEL6_PORT=8007

LOG_DIR="/home/nunes/Abacus/palimpzest/abacus-research/var/logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lotus

cd /home/nunes/lotus-HPC

wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}


# echo "[INFO] Starting vLLM embedding server"
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model nomic-ai/nomic-embed-text-v1 --task embed --port $EMBEDDING_PORT --trust-remote-code --max-model-len 8K &
# PIDEmbedding=$!
# wait_for_ready $EMBEDDING_PORT


echo "[INFO] Starting vLLM inference servers"
#checar se isso é seguro
export VLLM_SERVER_DEV_MODE=1

# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --enable-sleep-mode --model "$MODEL1_NAME" --port $MODEL1_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.85  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" &
# PID1=$!
# wait_for_ready $MODEL1_PORT
# echo "[INFO] Putting MODEL1 ($MODEL1_NAME) into Sleep Level 1"
# curl -s -X POST http://localhost:$MODEL1_PORT/sleep \
#      -H "Content-Type: application/json" \
#      -d '{"level": "2"}'


# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --enable-sleep-mode --model "$MODEL2_NAME" --port $MODEL2_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.85 2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
# PID2=$!
# wait_for_ready $MODEL2_PORT
# echo "[INFO] Putting MODEL2 ($MODEL2_NAME) into Sleep Level 1"
# curl -s -X POST http://localhost:$MODEL2_PORT/sleep \
#      -H "Content-Type: application/json" \
#      -d '{"level": "2"}'

# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --enable-sleep-mode --model "$MODEL3_NAME" --port $MODEL3_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.6  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL3_PORT.txt" &
# PID3=$!
# wait_for_ready $MODEL3_PORT
# echo "[INFO] Putting MODEL3 ($MODEL3_NAME) into Sleep Level 1"
# curl -s -X POST http://localhost:$MODEL3_PORT/sleep \
#      -H "Content-Type: application/json" \
#      -d '{"level": "2"}'

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --enable-sleep-mode --model "$MODEL4_NAME" --port $MODEL4_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.7  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL4_PORT.txt" &
PID4=$!
wait_for_ready $MODEL4_PORT
# echo "[INFO] Putting MODEL4 ($MODEL4_NAME) into Sleep Level 1"
# curl -s -X POST http://localhost:$MODEL4_PORT/sleep \
#      -H "Content-Type: application/json" \
#      -d '{"level": "2"}'

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --enable-sleep-mode --model "$MODEL5_NAME" --port $MODEL5_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.7  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL5_PORT.txt" &
PID5=$!
wait_for_ready $MODEL5_PORT
# echo "[INFO] Putting MODEL5 ($MODEL5_NAME) into Sleep Level 1"
# curl -s -X POST http://localhost:$MODEL5_PORT/sleep \
#      -H "Content-Type: application/json" \
#      -d '{"level": "2"}'

# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --enable-sleep-mode --model "$MODEL6_NAME" --port $MODEL6_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.60  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL6_PORT.txt" &
# PID6=$!
# wait_for_ready $MODEL6_PORT
# echo "[INFO] Putting MODEL6 ($MODEL6_NAME) into Sleep Level 1"
# curl -s -X POST http://localhost:$MODEL6_PORT/sleep \
#      -H "Content-Type: application/json" \
#      -d '{"level": "2"}'

# Wait for both to exit (Ctrl+C will trigger cleanup)

wait $PIDEmbedding $PID1 $PID4 $PID5 $PID6 #$PID2 $PID3