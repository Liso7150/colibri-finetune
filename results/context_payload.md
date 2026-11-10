# Compacted agent payload vs the old payload, live engine

Engine `qwen3.6-colibri` at `http://127.0.0.1:8197`, greedy, `max_tokens=8`, 2 alternating pairs. Same transcript, same system prompt and task; only the tool payload differs.

| Arm | chars | prompt tokens | median TTFT | median total |
|---|---|---|---|---|
| old | 25,068 | 9594 | 70.70 s | 71.03 s |
| new | 19,066 | 6995 | 51.14 s | 51.47 s |

**Delta:** 6,002 fewer characters (23.9 %), 2599 fewer prompt tokens, +19.56 s TTFT (+27.7 %), +19.56 s total per step.

Positive seconds are time saved by the compacted payload. TTFT is the prefill-bound figure; total includes the same number of decoded tokens in both arms.

## Trials

| pair | order | arm | chars | TTFT (s) | total (s) |
|---|---|---|---|---|---|
| 1 | old→new | old | 25,068 | 72.12 | 72.52 |
| 1 | old→new | new | 19,066 | 51.91 | 52.22 |
| 2 | new→old | new | 19,066 | 50.38 | 50.72 |
| 2 | new→old | old | 25,068 | 69.28 | 69.55 |

## Payload shape

`{"chars_before": 52543, "chars_after": 19413, "elided": 3, "deduped": 1, "budget_chars": 48000, "result_chars": 6000}`
