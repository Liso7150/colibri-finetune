# Engine tier profiles, same prompt and token budget

`D:/MEGAsync-sync/colibri-models/qwen36-35b`, `--ctx 32768`, 64 tokens, greedy. Higher is better; TTFT is prefill.

| Profile | Status | median tok/s | median TTFT | startup |
|---|---|---|---|---|
| pool 4 GB, gpu auto | ok | **17.31** | 0.48 s | 39.9 s |

Fastest: **pool 4 GB, gpu auto** at 17.31 tok/s (0.48 s TTFT).

NPU is not a candidate: `--xdna` is AMD-only and the Intel NPU has no path into this engine.
