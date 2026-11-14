# Result index

Every file here is evidence from a real run on the 8 GB RTX 4060 machine; the
scripts that produce them are in `../scripts/`. Re-run rather than edit.

| File | What it measures |
|---|---|
| `engine_vs_airllm.{json,md}` | Engine decode vs AirLLM alone (10.676 vs 0.0767 tok/s, 139×); footprint 15.42 vs 1.37 GiB |
| `engine_bench_run.log`, `engine_vs_airllm_engine.log` | the engine-side logs behind that comparison |
| `context_payload.{json,md}` | agent payload before/after core context compaction (25,068 → 19,066 chars, 9,594 → 6,995 prompt tokens, TTFT 70.703 → 51.141 s) |
| `preflight_qwen36_35b_ctx8192.json` | weights + KV memory budget from `fennec_core.preflight.plan` at ctx 8192 (22.777 GiB weights, 0.625 GiB KV, 23.402 GiB peak) |
| `tier_profiles.{json,md}`, `tier_profiles_lowram.{json,md}` | CPU-only vs Vulkan hybrid tier profiles, and the low-RAM pool sweep |
| `holdout_manifest_1.0.0.json`, `holdout_run_*.json`, `holdout_verdict.json` | frozen 60-case holdout and its zero-regression promotion verdict |
| `bench_best_finetune_*.json`, `finetune_verdict.json` | fine-tuned vs base comparison |
| `holdout_compare.png`, `holdout_compare_finetuned_vs_pre.txt` | the same comparison, plotted and diffed |
| `desktop_build.log`, `tier-*.log` | build and per-profile run logs |

The full engine-vs-AirLLM write-up is in `WORKSPACE.md` §8.1; the tier tables are
in §8.3 and §8.4.

## Historical paths (recorded, not current)

The archived engine/tier/holdout runs were recorded before the folders moved. Their
`model`/`path` fields and their `log` fields name absolute locations from that earlier
layout, on a drive and under directory names that no longer exist on this machine. The
models now live under the `models_root` in `~/.fennec/config.json` and the suite under its
own root; both are local paths, not published locations. The newer
`baseline_colibri_20261112.*` and `baseline_colibri_vk.*` already record the current root.

Those fields are deliberately left as measured. `engine.command` and `engine.log` in
`engine_vs_airllm.json` are verbatim records of the argv and log file that actually ran,
so rewriting only the `model` block would make the evidence contradict itself, and no
re-run has been done that would justify new numbers. Refresh the paths by re-running
`../scripts/bench_engine_vs_airllm.py` and `../scripts/bench_tier_profiles.py` against the
current config.

`preflight_qwen36_35b_ctx8192.json` is the exception: it was generated against the current
config, so its `model` field is a live local path. It is evidence, not a published location.

Affected: `engine_vs_airllm.{json,md}`, `tier_profiles.{json,md}`,
`tier_profiles_lowram.{json,md}`, `holdout_run_qwen36-35b-pre.json`,
`holdout_run_qwen36-35b_finetuned.json`.

The archived holdout reports also carry the old suite label `colibri-holdout-1.0.0`.
`bench_holdout.py` now emits `fennec-holdout-1.0.0` with an unchanged case fingerprint, so
`--compare` refuses these archives until they are re-run; the recorded verdicts stand as
evidence of the runs that produced them.
