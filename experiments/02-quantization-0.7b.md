# Experiment 2: Prompt Length Impact on Prefill vs Decode

## What I expected
Longer prompts should increase TTFT since prefill processes 
all input tokens before generating anything. Throughput (tok/s) 
should stay roughly the same since decode speed is independent 
of input length.

## Setup
Model: qwen2.5:0.5b and qwen2.5:7b-instruct-q4_K_M
Machine: MacBook Air M4 8GB
Prompts:
  medium → "Explain what a transformer is in 3 sentences."
  long   → "Explain the entire history of deep learning from 
             perceptrons to transformers, covering every major 
             architectural innovation."


## Numbers (7B model)
| Prompt  | Avg TTFT | Avg Tok/s |
|---------|----------|-----------|
| long    | 114.7ms  | 17.3      |

## What actually happened
On 0.5B, TTFT barely moved across prompt lengths. Throughput stayed almost perfectly stable. 
On 7B, TTFT jumped to 115ms and throughput crashed to 19.5 tok/s compared to 139 tok/s on 0.5B — but that's the model size effect, not prompt length.

The prompt length effect on TTFT was much smaller than expected on 0.5B. Apple's Neural Engine handles the prefill of a small model so efficiently that even a long prompt doesn't create visible latency.

## My explanation
The real prefill cost only becomes visible on larger models with longer contexts where the attention computation over all input tokens becomes substantial.

The 7B numbers confirm the memory bandwidth theory from experiment 1. 14x bigger model, roughly 7x slower throughput. 
The scaling isn't perfectly linear because unified memory partially absorbs the bandwidth pressure, but the trend is clear.

## Open questions
- At what model size does prompt length start visibly affecting 
  TTFT on Apple Silicon?
- On a GPU with discrete VRAM, would long prompts spike TTFT 
  more dramatically?
- Is the 7B throughput drop purely bandwidth or is thermal 
  throttling on the Air contributing?