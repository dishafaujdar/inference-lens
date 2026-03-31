# Inference Engineering 101 — The 20% That Drives 80%

Inference engineering is nothing but how one can decrease the gap between what a machine can actually perform and how it's performing now. Simple as that.

Each machine has a theoretical bar and most of the time it's underperforming. To achieve that bar, we do some engineering that majorly focuses on five pillars:

- **Parallelism** — splitting work across multiple cores or machines
- **Batching** — serving multiple requests in one pass
- **Memory management** — using what you have more efficiently
- **Kernel fusion** — combining operations so the hardware makes fewer trips
- **Model shrinking** — quantization, pruning, distillation

For each of these there are many inference engines (that perform the actual execution work) and serving frameworks out there, each pulling on these five pillars in their own way.

Which makes sense — you can't apply Apple's unified memory approach to NVIDIA GPUs, they have completely different hardware specs. You can't use the same scalability approach for a single laptop GPU and a thousand GPU datacenter. Different hardware has different loading dock specs. Different scales need different solutions.

That's why there are so many frameworks out there.

So when you encounter a new framework, the right question isn't *"what does this do?"* It's **"which of those 5 levers is this primarily pulling, and what constraint is it optimizing for?"**

---

## The Core Bottleneck: Memory-Bandwidth-Bound vs Compute-Bound

When it comes to LLMs specifically, every performance problem reduces to one of two bottlenecks.

To understand them, you need to know about two phases that happen during every LLM inference call.

**PREFILL** is when the model processes all your input tokens simultaneously — one giant matrix multiplication across all tokens at once. High arithmetic intensity. Lots of math per byte read. **Compute-bound.**

**DECODE** is when the model returns one token at a time, reading through all the model weights on every single step, doing almost no math, producing one token. Near-zero arithmetic intensity. **Memory-bandwidth-bound.**

The distinction is **how much math you do per byte you read** — not whether memory is involved.

---

## Memory-Bandwidth-Bound — The Delivery Guy Problem

Optimizations here: caching, batching, quantization.

Think of your GPU cores as a kitchen with 1000 chefs (CUDA cores) and your VRAM as a warehouse next door. The chefs are incredibly fast but they can only cook what the delivery guy brings them. The delivery guy is your memory bandwidth — he has a fixed speed regardless of how many chefs are waiting.

During decode, all 1000 chefs are standing idle waiting for the delivery guy to bring the next batch of weights. Adding more chefs (better GPU) doesn't help. You need:

- A faster delivery guy → higher memory bandwidth
- Smaller packages → quantization (fewer bytes per weight)
- Smarter delivery scheduling → batching (one trip serves multiple chefs simultaneously)

**It's not your GPU cores that are the bottleneck. It's how fast the weights can travel from memory to those cores.**

---

## Compute-Bound — The Kitchen Problem

Optimizations here: better GPUs, kernel fusion, tensor parallelism.

During prefill, the delivery guy is actually keeping up — he's bringing entire truckloads of tokens all at once (matrix multiplication across all input tokens simultaneously). Now the bottleneck flips. The chefs can't cook fast enough. All the ingredients are on the counter, the delivery guy is standing around waiting, but the kitchen is overwhelmed with orders.

That's compute-bound. Here:

- More chefs → better GPU, more CUDA cores, actually helps
- Bigger trucks → tensor parallelism across multiple GPUs, helps
- Teaching chefs to do multiple steps in one motion → kernel fusion, helps

**The warehouse isn't the problem anymore — the kitchen is.**

---

Once you get that flip, you'll intuitively understand why the same optimization that helps decode (quantization) does almost nothing for prefill — and why throwing a better GPU at a memory-bound problem is wasted money.

---

*Currently benchmarking this hands-on — running Ollama and vLLM experiments on real models, measuring TTFT, throughput, and KV cache behavior. Notes live in my [inference benchmarks repo](#).*