# Experiment 1: Quantization Impact on 0.5B Model

## What I expected
q4_K_M should be noticeably faster than fp16 since it's storing weights in half the bytes. Expected maybe 2x throughput difference.

## Setup
Model: qwen2.5:0.5b at three quantization levels
Machine: MacBook Air M4 8GB
Prompt: "Explain what a transformer is in 3 sentences."
Long_Prompt: "Explain the entire history of deep learning from perceptrons to transformers, covering every major architectural innovation."
Runs: 3 per model, discarding run 0 (cold)

## Numbers
| Model                        | Avg TTFT | Avg Tok/s |
|------------------------------|----------|-----------|
| qwen2.5:0.5b (default)       | 63.1ms   | 139.4     |
| qwen2.5:0.5b-instruct-q4_K_M | 64.8ms   | 139.4     |
| qwen2.5:0.5b-instruct-fp16   | 71.4ms   | 90.1      |

## What actually happened
q4_K_M and default showed zero meaningful difference. fp16 was ~35% slower in throughput but TTFT was nearly identical across 
all three. The gap between q4 and fp16 existed but was much smaller than expected.

## My explanation
Two things happening here. First, Apple Silicon has unified memory — CPU, GPU, and Neural Engine all share the same physical RAM pool with high bandwidth. There's no PCIe bottleneck like you'd see on a discrete GPU setup. Second, 0.5B is too small to create real memory pressure. The entire model fits comfortably in the memory pool regardless of quantization level. The fp16 gap shows up in throughput (more bytes moved per token) but not dramatically because the model is tiny.
The 35% throughput difference between q4 and fp16 IS the memory bandwidth constraint showing itself — fp16 moves.