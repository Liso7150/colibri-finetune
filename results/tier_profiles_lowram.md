# Engine tier profiles, same prompt and token budget

`D:/colibri-models/qwen36-35b`, `--ctx 8192`, 64 tokens, greedy. Higher is better; TTFT is prefill.

| Profile | Status | median tok/s | median TTFT | startup |
|---|---|---|---|---|
| pool 4 GB, gpu auto | ok | **14.82** | 0.53 s | 19.1 s |
| pool 4 GB + gpu 4 GB | ok | **14.93** | 0.53 s | 26.7 s |
| pool 6 GB + gpu 4 GB | ok | **14.66** | 0.55 s | 26.3 s |

Fastest: **pool 4 GB + gpu 4 GB** at 14.93 tok/s (0.53 s TTFT).

NPU is not a candidate: `--xdna` is AMD-only and the Intel NPU has no path into this engine.
