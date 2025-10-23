import re
import threading
import time
import csv
from datetime import datetime
from fastapi import FastAPI, Request
import httpx
import uvicorn
from threading import Lock
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

MODEL_ROUTES = {
    "Qwen/Qwen3-4B": ["http://localhost:8105", "http://localhost:8106"],
}

#passe os logs dos modelos que estiverem rodando para um arquivo e coloque aqui 
BACKEND_LOGS = {
    "http://localhost:8105": "./var/logs/saida_VLLM_8105.txt",
    "http://localhost:8106": "./var/logs/saida_VLLM_8106.txt",
}

BACKEND_METRICS = {
    url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
    for url in BACKEND_LOGS
}

BACKEND_QUEUE = {url: 0 for url in BACKEND_LOGS}
queue_lock = Lock()

LOG_PATTERN = re.compile(
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%"
)

metrics_lock = Lock()

# --- Métricas globais ---
LOG_FILE = "proxy_metrics.csv"
log_lock = Lock()
monitoring_active = False
first_request_time = None


def acquire_backend(candidates):
    """Pick backend with smallest queue and increment counter atomically."""
    with queue_lock:
        best_backend = min(candidates, key=lambda url: BACKEND_QUEUE[url])
        BACKEND_QUEUE[best_backend] += 1
    # print(f"[QUEUE] Assigned to {best_backend} (queue={current_q})")
    return best_backend


def release_backend(backend_url):
    """Decrement counter atomically."""
    with queue_lock:
        if BACKEND_QUEUE[backend_url] > 0:
            BACKEND_QUEUE[backend_url] -= 1
        # print(f"[QUEUE] Released {backend_url} (queue={BACKEND_QUEUE[backend_url]})")


def log_latency(latency, backend_url):
    """Salva a latência da requisição em CSV"""
    with log_lock:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), latency, backend_url])


def tail_log_file(path, pattern):
    """Yield new log lines matching pattern."""
    try:
        with open(path, "r") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                if pattern.search(line):
                    yield line
    except FileNotFoundError:
        print(f"[WARN] Log file {path} not found")
        while True:
            time.sleep(5)


def monitor_logs(backend_url, log_path):
    """Background thread that updates metrics from log."""
    for line in tail_log_file(log_path, LOG_PATTERN):
        match = LOG_PATTERN.search(line)
        if match:
            running, waiting, kv_cache = match.groups()
            with metrics_lock:
                BACKEND_METRICS[backend_url]["running"] = int(running)
                BACKEND_METRICS[backend_url]["waiting"] = int(waiting)
                BACKEND_METRICS[backend_url]["kv_cache"] = float(kv_cache)
                BACKEND_METRICS[backend_url]["last_updated"] = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start monitoring threads when app starts."""
    print("[INFO] Starting log monitoring threads...")
    for backend_url, path in BACKEND_LOGS.items():
        t = threading.Thread(target=monitor_logs, args=(backend_url, path), daemon=True)
        t.start()
    yield
    print("[INFO] Shutting down proxy...")


app = FastAPI(lifespan=lifespan)


def print_backend_status():
    """Print formatted metrics for all backends."""
    with metrics_lock:
        print("\n[STATUS] Current backend load:")
        print("-" * 70)
        for url, metrics in BACKEND_METRICS.items():
            port = url.split(":")[-1]
            running = metrics["running"]
            waiting = metrics["waiting"]
            kv = metrics["kv_cache"]
            updated_ago = time.time() - metrics["last_updated"]
            print(
                f"  • Port {port}: running={running:2d}, waiting={waiting:2d}, "
                f"KV={kv:5.1f}%, updated {updated_ago:4.1f}s ago"
            )
        print("-" * 70 + "\n")

@app.get("/queue")
async def get_queue_state():
    with queue_lock:
        return BACKEND_QUEUE.copy()

@app.get("/status")
async def get_status():
    """Return current backend metrics for visualization."""
    with metrics_lock:
        return BACKEND_METRICS


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    global monitoring_active, first_request_time

    start_time = time.time()
    body = await request.json()
    model = body.get("model")


    # Ativa o monitoramento somente quando chega a primeira requisição
    if not monitoring_active:
        monitoring_active = True
        first_request_time = start_time
        print(
            f"[METRICS] Monitoring started at {datetime.now().isoformat()} "
            "(first request received)"
        )

    if model not in MODEL_ROUTES:
        return {"error": f"Unknown model: {model}"}

    candidates = MODEL_ROUTES[model]

    with metrics_lock:
        current_metrics = {url: BACKEND_METRICS[url].copy() for url in candidates}

    print(f"\n[REQUEST] Received new request for model: {model}")
    print_backend_status()

    now = time.time()
    valid_backends = [
        url for url in candidates
        if now - current_metrics[url].get("last_updated", 0) < 30
    ]

    best_backend = acquire_backend(candidates)


    total_loads = {
        url: current_metrics[url]["running"] + current_metrics[url]["waiting"]
        for url in valid_backends
    }

    print(f"[INFO] Load summary: {total_loads}")
    print(f"[INFO] Selected backend for {model}: {best_backend}\n")

    target_url = f"{best_backend}/v1/chat/completions"
    
    try:
        async with httpx.AsyncClient(timeout=36000.0) as client:
            resp = await client.post(target_url, json=body)

        latency = time.time() - start_time
        log_latency(latency, best_backend)  # salva no CSV
        print(f"[METRIC] E2E latency = {latency:.3f}s | Backend = {best_backend}")

        return resp.json()
    except httpx.RequestError as e:
        latency = time.time() - start_time
        log_latency(latency, best_backend)
        print(f"[ERROR] Failed request ({latency:.3f}s): {e}")
        return {"error": f"Backend unavailable: {best_backend}"}
    
    finally:
        release_backend(best_backend)




@app.post("/v1/completions")
async def proxy_completion(request: Request):
    global monitoring_active, first_request_time

    start_time = time.time()
    body = await request.json()
    model = body.get("model")

    if not monitoring_active:
        monitoring_active = True
        first_request_time = start_time
        print(f"[METRICS] Monitoring started at {datetime.now().isoformat()} (first request)")

    if model not in MODEL_ROUTES:
        return {"error": f"Unknown model: {model}"}

    candidates = MODEL_ROUTES[model]


    best_backend = acquire_backend(candidates)

    print(f"[INFO] Selected backend for {model}: {best_backend}")


    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=36000.0) as client:
                async with client.stream("POST", f"{best_backend}/v1/completions", json=body) as backend_resp:
                    async for chunk in backend_resp.aiter_raw():
                        yield chunk
        finally:
            release_backend(best_backend)


    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",  # SSE from vLLM
        status_code=200,
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
