import requests
import time
import json

def benchmark(model: str, prompt: str, runs: int = 3):
    results = []
    
    for i in range(runs):
        start = time.perf_counter()
        first_token_time = None
        tokens_received = 0
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "options":{"temperature":0,"seed":42}},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                if not data.get("done"):
                    tokens_received += 1
        
        end = time.perf_counter()
        
        ttft = (first_token_time - start) * 1000
        decode_time = end - first_token_time
        tps = tokens_received / decode_time
        
        # discard run 0, it's cold
        if i > 0:
            results.append({
                "ttft_ms": round(ttft, 1),
                "tok_s": round(tps, 1),
                "tokens": tokens_received
            })
        
        print(f"Run {i}: TTFT={ttft:.0f}ms | {tps:.1f} tok/s | {tokens_received} tokens")
    
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"\nAVG → TTFT: {avg_ttft:.1f}ms | Throughput: {avg_tps:.1f} tok/s\n")
    return avg_ttft, avg_tps


prompt = "Explain the entire history of deep learning from perceptrons to transformers, covering every major architectural innovation."

models = [
    # "qwen2.5:0.5b",
    # "qwen2.5:0.5b-instruct-q4_K_M",
    # "qwen2.5:0.5b-instruct-fp16",
    "qwen2.5:7b-instruct-q4_K_M"
]

for model in models:
    print(f"=== {model} ===")
    benchmark(model, prompt)