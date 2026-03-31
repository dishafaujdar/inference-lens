import asyncio
import aiohttp
import time

async def single_request(session, model, prompt):
    start = time.perf_counter()
    first_token = None
    tokens = 0
    
    async with session.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, 
              "options": {"temperature": 0, "seed": 42}}
    ) as resp:
        async for line in resp.content:
            if line:
                import json
                data = json.loads(line)
                if first_token is None:
                    first_token = time.perf_counter()
                if not data.get("done"):
                    tokens += 1
    
    end = time.perf_counter()
    return {
        "ttft_ms": (first_token - start) * 1000,
        "tok_s": tokens / (end - first_token),
        "tokens": tokens
    }

async def concurrent_benchmark(model, prompt, n_concurrent):
    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        tasks = [single_request(session, model, prompt) 
                 for _ in range(n_concurrent)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start
    
    total_tokens = sum(r["tokens"] for r in results)
    print(f"\n{n_concurrent} concurrent requests:")
    print(f"Total throughput: {total_tokens/total_time:.1f} tok/s")
    print(f"Avg TTFT: {sum(r['ttft_ms'] for r in results)/len(results):.1f}ms")
    print(f"Avg per-request tok/s: {sum(r['tok_s'] for r in results)/len(results):.1f}")

asyncio.run(concurrent_benchmark("qwen2.5:7b-instruct-q4_K_M", "Explain transformers", 1))
asyncio.run(concurrent_benchmark("qwen2.5:7b-instruct-q4_K_M", "Explain transformers", 4))
asyncio.run(concurrent_benchmark("qwen2.5:7b-instruct-q4_K_M", "Explain transformers", 8))