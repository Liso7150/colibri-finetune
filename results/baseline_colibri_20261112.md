# Engine tier profiles, same prompt and token budget

`D:/MEGAsync-sync/colibri-models/qwen36-35b`, `--ctx 32768`, 64 tokens, greedy. Higher is better; TTFT is prefill.

| Profile | Status | median tok/s | median TTFT | startup |
|---|---|---|---|---|
| pool 4 GB, gpu auto | error | — | — | — |

NPU is not a candidate: `--xdna` is AMD-only and the Intel NPU has no path into this engine.
