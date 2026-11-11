# Result index

Every file here is evidence from a real run on the 8 GB RTX 4060 machine; the
scripts that produce them are in `../scripts/`. Re-run rather than edit.

| File | What it measures |
|---|---|
| `engine_vs_airllm.{json,md}` | Engine decode vs AirLLM alone (10.676 vs 0.0767 tok/s, 139×); footprint 15.42 vs 1.37 GiB |
| `engine_bench_run.log`, `engine_vs_airllm_engine.log` | the engine-side logs behind that comparison |
| `context_payload.{json,md}` | agent payload before/after core context compaction (28,117 → 18,755 chars per step) |
| `tier_profiles.{json,md}`, `tier_profiles_lowram.{json,md}` | CPU-only vs Vulkan hybrid tier profiles, and the low-RAM pool sweep |
| `holdout_manifest_1.0.0.json`, `holdout_run_*.json`, `holdout_verdict.json` | frozen 60-case holdout and its zero-regression promotion verdict |
| `bench_best_finetune_*.json`, `finetune_verdict.json` | fine-tuned vs base comparison |
| `holdout_compare.png`, `holdout_compare_finetuned_vs_pre.txt` | the same comparison, plotted and diffed |
| `desktop_build.log`, `tier-*.log` | build and per-profile run logs |

The full engine-vs-AirLLM write-up is in `WORKSPACE.md` §8.1; the tier tables are
in §8.3 and §8.4.
