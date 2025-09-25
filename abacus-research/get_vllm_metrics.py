import requests
import concurrent.futures
from time import time, sleep
from prometheus_client.parser import text_string_to_metric_families
from collections import defaultdict
import sys

url_base = "http://127.0.0.1:8001"

# For throughput calculations
avg_tokens_prompt_prev = 0
avg_tokens_gen_prev = 0

### Metrics
## Obs: some also have _created metrics, showing creation time
# vllm:num_requests_running
# vllm:num_requests_waiting
# vllm:gpu_cache_usage_perc
# vllm:gpu_prefix_cache_queries
# vllm:gpu_prefix_cache_hits
# vllm:kv_cache_usage_perc
# vllm:prefix_cache_queries
# vllm:prefix_cache_hits
# vllm:num_preemptions
# vllm:prompt_tokens
# vllm:generation_tokens
# vllm:request_success
# vllm:request_prompt_tokens
# vllm:request_generation_tokens
# vllm:iteration_tokens_total
# vllm:request_max_num_generation_tokens
# vllm:request_params_n
# vllm:request_params_n_created
# vllm:request_params_max_tokens
# vllm:time_to_first_token_seconds
# vllm:time_per_output_token_seconds
# vllm:e2e_request_latency_seconds
# vllm:request_queue_time_seconds
# vllm:request_inference_time_seconds
# vllm:request_prefill_time_seconds
# vllm:request_decode_time_seconds
# vllm:cache_config_info


# Retry logic to wait until the server is up
def wait_for_server(timeout=600, retry_interval=1):
    start_time = time()
    while time() - start_time < timeout:
        try:
            response = requests.get(url_base + "/health")
            if response.status_code < 500:
                print("Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass  # Server not up yet
        print(f"Waiting for server at {url_base}...")
        sleep(retry_interval)
    print("Timeout waiting for server.")
    return False

def get_vllm_metrics_json(should_get_hist=False):
    response = requests.get(url_base + '/metrics')
    response.raise_for_status()  # Raise error if request failed
    raw_metrics = response.text

    metrics = {}

    for family in text_string_to_metric_families(raw_metrics):
        #print(family)
        f_name = family.name

        data = defaultdict(lambda: [])
        for sample in family.samples:
            is_not_hist = 'bucket' not in sample.name
            if is_not_hist or should_get_hist:
                data[sample.name].append((sample.labels, sample.value))
            
        #    metric = {
        #        "name": sample.name,
        #        "labels": sample.labels,
        #        "value": sample.value,
        #        "type": family.type,
        #        "help": family.documentation
        #    }
            #metrics_json.append(metric)
        final_data = {}
        for s_name, samples in data.items():
            if len(samples) == 1:
                label, value = samples[0]
                final_data[s_name] = value
            else:
                final_data[s_name] = samples
        metrics[f_name] = final_data
    return metrics

def print_metrics(interval):
    global avg_tokens_prompt_prev
    global avg_tokens_gen_prev

    metrics = get_vllm_metrics_json()

    def get_avg(name):
        if metrics[name][name + '_count'] == 0:
            return 0
        return metrics[name][name + '_sum'] / metrics[name][name + '_count']

    # avg_latency = get_avg('vllm:e2e_request_latency_seconds')
    # prefix_name = 'vllm:prefix_cache_'
    # if metrics[prefix_name + 'queries'][prefix_name + 'queries_total'] > 0:
    #     prefix_hit_rate = metrics[prefix_name + 'hits'][prefix_name + 'hits_total'] / metrics[prefix_name + 'queries'][prefix_name + 'queries_total']
    # else:
    #     prefix_hit_rate = 0

    req_run_name = 'vllm:num_requests_running'
    num_req_run = metrics[req_run_name][req_run_name]

    req_wait_name = 'vllm:num_requests_waiting'
    num_req_wait = metrics[req_wait_name][req_wait_name]

    gpu_cache_name = 'vllm:gpu_cache_usage_perc'
    gpu_cache_use = metrics[gpu_cache_name][gpu_cache_name]

    # kv_cache_name = 'vllm:kv_cache_usage_perc'
    # kv_cache_use = metrics[kv_cache_name][kv_cache_name]

    n_preempt_name = 'vllm:num_preemptions'
    num_preemptions = metrics[n_preempt_name][n_preempt_name + '_total']

    # avg_ttft = get_avg('vllm:time_to_first_token_seconds')
    # avg_tpot = get_avg('vllm:time_per_output_token_seconds')

    # avg_req_queue_time = get_avg('vllm:request_queue_time_seconds')
    # avg_req_inf_time = get_avg('vllm:request_inference_time_seconds')
    # avg_req_pref_time = get_avg('vllm:request_prefill_time_seconds')
    # avg_req_dec_time = get_avg('vllm:request_decode_time_seconds')

    tokens_prompt_name = 'vllm:prompt_tokens'
    avg_tokens_prompt = metrics[tokens_prompt_name][tokens_prompt_name + '_total']
    tokens_gen_name = 'vllm:generation_tokens'
    avg_tokens_gen = metrics[tokens_gen_name][tokens_gen_name + '_total']

    prompt_thpt = avg_tokens_prompt - avg_tokens_prompt_prev
    prompt_thpt /= interval
    avg_tokens_prompt_prev = avg_tokens_prompt
    gen_thpt = avg_tokens_gen - avg_tokens_gen_prev
    gen_thpt /= interval
    avg_tokens_gen_prev = avg_tokens_gen
 
    #avg_ = get_avg('vllm:')


    print(f"|{num_req_run}|{num_req_wait}|{gpu_cache_use}|{num_preemptions}|{avg_tokens_prompt}|{avg_tokens_gen}|{prompt_thpt}|{gen_thpt}")

def main():
    assert len(sys.argv) == 2, "Must set the logging interval in seconds"

    wait_for_server()
    print(f"avg_latency|prefix_hit_rate|num_req_run|num_req_wait|gpu_cache_use|kv_cache_use|num_preemptions|avg_ttft|avg_tpot|avg_req_queue_time|avg_req_inf_time|avg_req_pref_time|avg_req_dec_time|avg_tokens_prompt|avg_tokens_gen|prompt_thpt|gen_thpt")
    while True:
        print_metrics(int(sys.argv[1]))
        sleep(int(sys.argv[1]))


if __name__ == '__main__':
    main()

