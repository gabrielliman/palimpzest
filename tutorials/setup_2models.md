# Executar dois modelos em uma gpu

1. O parâmetro --port foi adicionado, pois assim é possível executar dois servidores distintos em uma mesma gpu.

2. Além de o parâmetro --gpu-memory-utilization ter sido alterado de 0.85 para 0.4, o modelo executado possui menos parâmetros.

3. Comando para subir dois servidores de llm na mesma gpu (sempre escolher o enviornment certo):
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype bfloat16 \
    --max-model-len 10000 \
    --port 8000 \
    --gpu-memory-utilization 0.4 \
    2>&1 | tee ./saida_VLLM_8000.txt &

    sleep 30

    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bfloat16 \
        --max-model-len 10000 \
        --port 8001 \
        --gpu-memory-utilization 0.4 \
        2>&1 | tee ./saida_VLLM_8001.txt &
