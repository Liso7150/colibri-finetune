# Engine tier profiles, same prompt and token budget

`D:/colibri-models/qwen36-35b`, `--ctx 8192`, 64 tokens, greedy. Higher is better; TTFT is prefill.

| Profile | Status | median tok/s | median TTFT | startup |
|---|---|---|---|---|
| cpu only (no Vulkan) | ok | **8.38** | 0.69 s | 9.9 s |
| cpu+gpu hybrid | ok | **15.75** | 1.17 s | 34.8 s |
| cpu+gpu, experimental-fast | ok | **15.81** | 1.2 s | 39.5 s |

Fastest: **cpu+gpu, experimental-fast** at 15.81 tok/s (1.2 s TTFT).

NPU is not a candidate: `--xdna` is AMD-only and the Intel NPU has no path into this engine.

## Utilisation during the hybrid run

One 128-token generation at 14.83 tok/s on the hybrid config, sampled every 1.5 s:
GPU 45-63 % (median 59 %), VRAM 6.4 GiB, CPU load 17-48 % (median 23 %). Both
tiers work on every token, but neither saturates, so the remaining latency is
expert streaming and synchronisation rather than idle hardware.

## Caveats

- Absolute throughput moves a lot between engine restarts: the same reference
  config measured 15.75 tok/s in the paired run and 6.12 tok/s immediately after
  a restart, while the OS reclaimed the previous engine's 13 GiB. Compare configs
  inside one run, never across restarts.
- `--vram` is a CUDA-build flag; this Vulkan engine exits with "`--vram` needs the
  CUDA build", so the VRAM budget stays the engine's own choice here.
- No NPU row: `--xdna` is AMD-only and the Intel NPU has no path into the engine.
