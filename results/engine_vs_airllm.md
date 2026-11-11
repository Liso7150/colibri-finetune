# Colibri engine vs AirLLM alone

Generated 2026-11-11 08:10:31 on Windows-11-10.0.26300-SP0 (22 threads, Python 3.12.10).

- Engine model: `D:/colibri-models/qwen36-35b`
- AirLLM model: `D:/colibri-models/Qwen3.6-35B-A3B-bf16`
- Prompts: 3, 32 new tokens each, greedy/stream
- AirLLM budget: 4 new tokens (it streams every layer per token, so a full sweep can take hours)

| Backend | Status | Load s | Median TTFT s | Decode tok/s | End-to-end tok/s | Peak RAM GiB | Peak VRAM GiB |
|---|---|---|---|---|---|---|---|
| Colibri engine | ok | 38.14 | 0.485 | 10.676 | 6.686 | 15.42 | 6.27 |
| AirLLM only | ok | 2.39 | n/a | 0.0767 | 0.054 | 1.37 | 0.36 |

Counting: the engine reports real `usage.completion_tokens`; its decode rate excludes
time-to-first-token. AirLLM returns the whole sequence in one call, so its end-to-end rate
includes prefill and it has no separate TTFT. To compare like with like, AirLLM's decode
rate subtracts the timed 1-token warm-up (prefill + 1 decode) from the timed run (prefill +
N decode) and divides by N-1. Both phases run that warm-up first. Peak RAM is the sampled
resident set of the whole process tree.

- Colibri engine backend: `vulkan`
- AirLLM only: airllm on `cpu`, torch 2.15.0+cpu, cuda=False
- AirLLM only adapter: explicit layer names for model.language_model.layers
- AirLLM only chat template: explicit Qwen fallback

**Verdict:** keep the Colibri engine - 10.676 tok/s decode against 0.0767 tok/s for AirLLM alone (139x), both measured decode-only.

## Per-prompt detail

### Colibri engine
- `8.0` tok/s decode, 0.531s for 1 tok, ttft 0.406s - pong
- `10.676` tok/s decode, 1.047s for 7 tok, ttft 0.485s - The capital of France is Paris.
- `12.241` tok/s decode, 1.562s for 14 tok, ttft 0.5s - ```python reversed_list = my_list[::-1] ```

### AirLLM only
- `0.0796` tok/s decode, 73.547s for 4 tok, ttft n/as - <think> Here's
- `0.0751` tok/s decode, 73.594s for 4 tok, ttft n/as - <think>  </think>
- `0.0767` tok/s decode, 74.281s for 4 tok, ttft n/as - <think> Here's

