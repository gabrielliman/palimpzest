import re
import threading
import time
from fastapi import FastAPI, Request
import httpx
import uvicorn
import asyncio
from threading import Lock
from contextlib import asynccontextmanager

# Map Hugging Face model names -> vLLM backends
MODEL_ROUTES = {
    "Qwen/Qwen2.5-1.5B-Instruct": ["http://localhost:8002"],
    "meta-llama/Llama-3.1-8B-Instruct": ["http://localhost:8003"],
    # "Qwen/Qwen2.5-3B-Instruct": ["http://localhost:8004"],
}

# Configuração de uso de GPU por backend
BACKEND_GPU_USAGE = {
    "http://localhost:8002": {"active": 40, "sleep": 4},   # Qwen-1.5B
    "http://localhost:8003": {"active": 85, "sleep": 5},   # LLaMA-8B
    # "http://localhost:8004": {"active": 60, "sleep": 5}, # outros modelos
}

# Map backends -> log file path
BACKEND_LOGS = {
    "http://localhost:8002": "/home/nunes/Abacus/palimpzest/abacus-research/var/logs/saida_VLLM_8002.txt",
    "http://localhost:8003": "/home/nunes/Abacus/palimpzest/abacus-research/var/logs/saida_VLLM_8003.txt",
    "http://localhost:8004": "/home/nunes/Abacus/palimpzest/abacus-research/var/logs/saida_VLLM_8004.txt",
}

# Metrics dictionary with thread safety
BACKEND_METRICS = {
    url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
    for url in BACKEND_LOGS
}
metrics_lock = Lock()

BACKEND_STATE = {url: "unknown" for url in BACKEND_LOGS}
state_lock = Lock()

LOG_PATTERN = re.compile(
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%"
)

LAST_USED = {url: 0 for url in BACKEND_GPU_USAGE}

MODEL_QUEUES = {model: asyncio.Queue() for model in MODEL_ROUTES}
QUEUE_METRICS = {model: {"size": 0} for model in MODEL_ROUTES}
queue_lock = Lock()

BACKEND_MAX_WORKERS = {
    "http://localhost:8002": 30,   # Qwen-1.5B suporta menos workers
    "http://localhost:8003": 5,   # LLaMA-8B suporta mais
    # "http://localhost:8004": 4,   # outros
}




def mark_used(url: str):
    LAST_USED[url] = time.time()


def get_current_gpu_usage():
    total = 0
    with state_lock:
        for url, state in BACKEND_STATE.items():
            usage = 0
            if state in ("wake", "working"):
                usage = BACKEND_GPU_USAGE[url]["active"]
            elif state == "sleep":
                usage = BACKEND_GPU_USAGE[url]["sleep"]
            total += usage
    return total

def tail_log_file(path, pattern):
    """Generator that yields new lines from log file"""
    try:
        with open(path, "r") as f:
            f.seek(0, 2)  # go to end of file
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                if pattern.search(line):
                    yield line
    except FileNotFoundError:
        print(f"Warning: Log file {path} not found")
        while True:
            time.sleep(5)  # Keep thread alive but don't process

def monitor_logs(backend_url, log_path):
    """Thread that updates BACKEND_METRICS from log"""
    for line in tail_log_file(log_path, LOG_PATTERN):
        match = LOG_PATTERN.search(line)
        if match:
            running, waiting, kv_cache = match.groups()
            with metrics_lock:
                BACKEND_METRICS[backend_url]["running"] = int(running)
                BACKEND_METRICS[backend_url]["waiting"] = int(waiting)
                BACKEND_METRICS[backend_url]["kv_cache"] = float(kv_cache)
                BACKEND_METRICS[backend_url]["last_updated"] = time.time()
                
def update_backend_state_from_queue(model: str):
    backend_url = MODEL_ROUTES[model][0]
    with queue_lock:
        queue_size = QUEUE_METRICS[model]["size"]

    with state_lock:
        if queue_size > 0 and BACKEND_STATE[backend_url]=="wake":
            BACKEND_STATE[backend_url] = "working"
        else:
            if BACKEND_STATE[backend_url] == "working":
                asyncio.create_task(try_transition_to_wake(model))



async def try_transition_to_wake(model: str):
    """Verifica após cooldown se pode voltar de working -> wake"""
    await asyncio.sleep(5)
    backend_url = MODEL_ROUTES[model][0]

    with metrics_lock:
        running = BACKEND_METRICS[backend_url]["running"]
        waiting = BACKEND_METRICS[backend_url]["waiting"]
    with queue_lock:
        queue_empty = QUEUE_METRICS[model]["size"] == 0

    if queue_empty and running == 0 and waiting == 0:
        with state_lock:
            if BACKEND_STATE[backend_url] == "working":
                BACKEND_STATE[backend_url] = "wake"
                print(f"[INFO] Backend {backend_url} voltou para estado wake (fila vazia e ocioso).")


async def check_backend_state(backend_url: str):
    """Checa estado inicial de um backend"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{backend_url}/health")
            if r.status_code == 200:
                # Servidor rodando, verificar se está dormindo
                s = await client.get(f"{backend_url}/is_sleeping")
                if s.status_code == 200 and s.json().get("is_sleeping", False):
                    state = "sleep"
                else:
                    # Se não está dormindo, pode ser wake ou working
                    with metrics_lock:
                        running = BACKEND_METRICS[backend_url]["running"]
                        waiting = BACKEND_METRICS[backend_url]["waiting"]

                    if running > 0 or waiting > 0:
                        state = "working"
                    else:
                        state = "wake"
            else:
                state = "dead"
    except Exception:
        state = "dead"

    with state_lock:
        BACKEND_STATE[backend_url] = state

async def wake_backend_if_needed(url: str):
    """Se o backend estiver dormindo, manda wake_up (só pesos) e espera ficar pronto"""
    with state_lock:
        state = BACKEND_STATE.get(url, "dead")

    if state in ("wake", "working"):
        return True
    elif state == "dead":
        print(f"[ERROR] Backend {url} está morto, não é possível acordar")
        return False
    elif state != "sleep":
        print(f"[WARN] Backend {url} está em estado {state}, não pode acordar")
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            print(f"[INFO] Acordando backend {url} (somente pesos)...")
            await client.post(f"{url}/wake_up", json={"tags": ["weights"]})

        # Poll até ficar pronto
        for _ in range(30):  # até 30s
            async with httpx.AsyncClient(timeout=5.0) as client:
                #trocar por is sleeping
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    await asyncio.sleep(2)
                    with state_lock:
                        BACKEND_STATE[url] = "wake"
                    print(f"[INFO] Backend {url} acordou com sucesso e está pronto")
                    return True
            await asyncio.sleep(1)

        print(f"[ERROR] Timeout ao tentar acordar {url}")
        return False
    except Exception as e:
        print(f"[ERROR] Falha ao acordar backend {url}: {e}")
        return False





async def put_backend_to_sleep(url: str):
    with state_lock:
        if BACKEND_STATE.get(url) != "wake":
            return False

    # Descobre modelo associado ao backend
    model = None
    for m, backends in MODEL_ROUTES.items():
        if url in backends:
            model = m
            break

    # 🔒 Checagem atômica: fila + métricas + estado
    with queue_lock, metrics_lock, state_lock:
        queue_empty = (QUEUE_METRICS[model]["size"] == 0) if model else True
        running = BACKEND_METRICS[url]["running"]
        waiting = BACKEND_METRICS[url]["waiting"]
        state_ok = BACKEND_STATE.get(url) == "wake"

        if not (state_ok and queue_empty and running == 0 and waiting == 0):
            # print(f"[INFO] Backend {url} não pode dormir ainda "
            #       f"(queue={not queue_empty}, running={running}, waiting={waiting})")
            return False

    # Só chega aqui se passou pela checagem atômica
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{url}/sleep", json={"level": "1"})

        for _ in range(30):
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{url}/is_sleeping")
                if r.status_code == 200 and r.json().get("is_sleeping", False):
                    with state_lock:
                        BACKEND_STATE[url] = "sleep"
                    usage = get_current_gpu_usage()
                    print(f"[INFO] Backend {url} entrou em sleep. GPU usage agora: {usage}%")
                    return True
            await asyncio.sleep(1)

        print(f"[ERROR] Timeout ao colocar {url} em sleep")
        return False
    except Exception as e:
        print(f"[ERROR] Falha ao colocar {url} em sleep: {e}")
        return False




# working
# wake
# sleep
# dead

async def periodic_status_printer():
    """A cada 3 segundos mostra estado, métricas e fila de cada modelo"""
    while True:
        print("\n[STATUS REPORT]")
        with state_lock:
            states_copy = BACKEND_STATE.copy()
        with metrics_lock:
            metrics_copy = {url: BACKEND_METRICS[url].copy() for url in BACKEND_METRICS}

        for url, state in states_copy.items():
            await check_backend_state(url)
            metrics = metrics_copy.get(url, {})
            print(f"Backend {url} -> State: {state}, "
                  f"Running: {metrics.get('running', 0)}, "
                  f"Waiting: {metrics.get('waiting', 0)}, "
                  f"KV Cache: {metrics.get('kv_cache', 0.0)}%, "
                  f"Last updated: {time.time() - metrics.get('last_updated', 0):.1f}s ago")

        for model in MODEL_ROUTES:
            with queue_lock:
                size = QUEUE_METRICS[model]["size"]
            print(f"Model {model} -> Queue size: {size}")

        await asyncio.sleep(3)



async def cycle_put_backend_to_sleep(exclude: str, needed: int):
    """
    Tenta colocar algum backend em sleep para liberar GPU.
    - exclude: backend que não pode ser colocado para dormir
    - needed: percentual de GPU que precisamos liberar
    """
    while True:
        # Seleciona candidatos que estão em wake (mas não o excluído)
        for url in BACKEND_STATE.keys():
            await check_backend_state(url)

        candidates = [u for u, s in BACKEND_STATE.items() if s == "wake" and u != exclude]

        if not candidates:
            await asyncio.sleep(5)
            continue

        # Ordena por último uso (se não quiser essa lógica, pode só iterar em ordem)
        candidates = sorted(candidates, key=lambda u: LAST_USED.get(u, 0))

        for url in candidates:
            ok = await put_backend_to_sleep(url)
            if ok:
                usage_now = get_current_gpu_usage()
                projected = usage_now - BACKEND_GPU_USAGE[exclude]["sleep"] + needed
                print(f"[DEBUG] Após dormir {url}, GPU usage: {usage_now}%. Projeção: {projected}%")
                if projected <= 95:
                    return True

        print("[INFO] Nenhum backend pôde dormir nesse ciclo. Tentando novamente em 10s...")
        print(BACKEND_STATE)
        await asyncio.sleep(10)


gpu_lock = asyncio.Lock()

async def ensure_wakeup_with_budget(url: str):
    with state_lock:
        current_state = BACKEND_STATE.get(url, "dead")

    # 🚀 Se já está acordado ou trabalhando, não precisa lock
    if current_state in ("wake", "working"):
        return True

    # Para acordar, usa o lock global (evita corridas)
    async with gpu_lock:
        needed = BACKEND_GPU_USAGE[url]["active"]
        usage_now = get_current_gpu_usage()
        projected = usage_now - BACKEND_GPU_USAGE[url]["sleep"] + needed

        # print(f"[DEBUG] GPU usage atual: {usage_now}%. Projeção ao acordar {url}: {projected}%")

        while projected > 95:
            # print(f"[INFO] GPU projected {projected}%, tentando liberar espaço...")
            ok = await cycle_put_backend_to_sleep(exclude=url, needed=needed)
            if not ok:
                return False
            usage_now = get_current_gpu_usage()
            projected = usage_now - BACKEND_GPU_USAGE[url]["sleep"] + needed
            # print(f"[DEBUG] Nova projeção: {projected}%")

        ok = await wake_backend_if_needed(url)
        if ok:
            mark_used(url)
            usage_now = get_current_gpu_usage()
            print(f"[INFO] Backend {url} acordado. GPU usage agora: {usage_now}%")
        return ok

    
async def _send_request(backend_url, body, future):
    target_url = f"{backend_url}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            resp = await client.post(target_url, json=body)
            result = resp.json()
    except Exception as e:
        result = {"error": f"Falha ao enviar para {backend_url}: {e}"}

    if not future.done():
        future.set_result(result)




async def model_worker(model: str):
    queue = MODEL_QUEUES[model]
    backends = MODEL_ROUTES[model]
    while True:
        body, future = await queue.get()
        best_backend = backends[0]

        ok = await ensure_wakeup_with_budget(best_backend)
        if not ok:
            if not future.done():
                future.set_result({"error": f"Backend {best_backend} indisponível"})
            # reenfileira sem mexer no contador
            await queue.put((body, future))
            update_backend_state_from_queue(model)
            queue.task_done()
            continue

        try:
            # 🚀 Worker envia e aguarda terminar
            await _send_request(best_backend, body, future)
        finally:
            with queue_lock:
                QUEUE_METRICS[model]["size"] -= 1
            queue.task_done()
            update_backend_state_from_queue(model)







from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Checking initial backend states...")
    tasks = [check_backend_state(url) for url in BACKEND_LOGS]
    await asyncio.gather(*tasks)
    with state_lock:
        print("[INFO] Initial BACKEND_STATE:", BACKEND_STATE)

    for url, path in BACKEND_LOGS.items():
        t = threading.Thread(target=monitor_logs, args=(url, path), daemon=True)
        t.start()

    # 🚀 Criar workers exclusivos para cada modelo
    for model, backends in MODEL_ROUTES.items():
        for backend_url in backends:
            max_workers = BACKEND_MAX_WORKERS.get(backend_url, 2)  # fallback = 2
            for i in range(max_workers):
                asyncio.create_task(model_worker(model))

    asyncio.create_task(periodic_status_printer())



    yield
    print("[INFO] Shutting down...")

app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.json()
    model = body.get("model")

    if model not in MODEL_ROUTES:
        return {"error": f"Unknown model: {model}"}

    candidates = MODEL_ROUTES[model]
    best_backend = candidates[0]  # por enquanto só 1

    future = asyncio.get_event_loop().create_future()
    await MODEL_QUEUES[model].put((body, future))
    # print(f"[DEBUG] Enfileirando requisição no modelo: {model}")
    with queue_lock:
        QUEUE_METRICS[model]["size"] += 1
    
    update_backend_state_from_queue(model)
    mark_used(best_backend)

    return await future


# Versão antiga sem filas e workers
# @app.post("/v1/chat/completions")
# async def proxy_chat(request: Request):
#     body = await request.json()
#     model = body.get("model")
    
#     if model not in MODEL_ROUTES:
#         return {"error": f"Unknown model: {model}"}

#     candidates = MODEL_ROUTES[model]
    
#     # Get current metrics with thread safety
#     with metrics_lock:
#         current_metrics = {url: BACKEND_METRICS[url].copy() for url in candidates}
    
#     # Filter out backends that haven't been updated in the last 30 seconds
#     current_time = time.time()
#     valid_backends = [
#         url for url in candidates 
#         if current_time - current_metrics[url].get("last_updated", 0) < 30
#     ]
    
#     if not valid_backends:
#         # Fallback to first candidate if no valid metrics
#         best_backend = candidates[0]
#         print(f"[WARNING] No recent metrics, using fallback: {best_backend}")
#     else:
#         # Select backend with lowest total load (running + waiting)
#         best_backend = min(
#             valid_backends,
#             key=lambda url: current_metrics[url]["running"] + current_metrics[url]["waiting"]
#         )
#     mark_used(best_backend)
#     ok = await ensure_wakeup_with_budget(best_backend)
#     if not ok:
#         return {"error": f"Backend {best_backend} indisponível"}
    
#     print(f"[DEBUG] Request for model {model}")
#     for url in candidates:
#         metrics = current_metrics[url]
#         status = "VALID" if url in valid_backends else "STALE"
#         print(f"[DEBUG] Backend {url} ({status}) -> running: {metrics['running']}, waiting: {metrics['waiting']}, updated: {current_time - metrics.get('last_updated', 0):.1f}s ago")
#     print(f"[DEBUG] Selected backend: {best_backend}")

#     # Forward request
#     target_url = f"{best_backend}/v1/chat/completions"
#     try:
#         async with httpx.AsyncClient(timeout=36000.0) as client:
#             print(f"[DEBUG] Forwarding request to {target_url}")
#             resp = await client.post(target_url, json=body)
#         return resp.json()
#     except httpx.RequestError as e:
#         print(f"[ERROR] Failed to forward request to {best_backend}: {e}")
#         return {"error": f"Backend unavailable: {best_backend}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)