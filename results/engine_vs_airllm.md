# Colibri engine vs AirLLM alone

Generated 2026-11-10 20:46:21 on Windows-11-10.0.26300-SP0 (22 threads, Python 3.14.7).

- Engine model: `D:/colibri-models/qwen36-35b`
- AirLLM model: `D:/colibri-models/Qwen3.6-35B-A3B-bf16`
- Prompts: 3, 24 new tokens each, greedy/stream

| Backend | Status | Median TTFT | Median tokens/s | Notes |
|---|---|---|---|---|
| Colibri engine | ok | 0.862 | 4.902 | vulkan |
| AirLLM only | error | n/a | n/a | load failed: ValueError: invalid literal for int() with base 10: 'layers' |

## Per-prompt detail

### Colibri engine
- `4.902` tok/s, 0.808s, ttft 0.604s - pong
- `5.988` tok/s, 1.703s, ttft 0.868s - The capital of France is Paris.
- `3.39` tok/s, 2.042s, ttft 0.862s - ```python reversed_list = my_list[::-1] ```

### AirLLM only

