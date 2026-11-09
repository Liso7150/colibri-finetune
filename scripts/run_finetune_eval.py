"""Post-training chain: merge LoRA -> convert int4 gs64 -> serve -> bench -> graph.

Run after airllm_train_qwen36.py finishes:
  .venv/Scripts/python.exe scripts/run_finetune_eval.py
Logs to stdout (redirect to results/finetune_eval.log). Promotion is a separate,
manual step after reviewing results/finetune_verdict.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
RESULTS = ROOT / "results"
MODELS = Path("D:/colibri-models")
ADAPTER = MODELS / "qwen36-lora-adapter.pt"
MERGED = MODELS / "Qwen3.6-35B-A3B-bf16-lora"
CANDIDATE = MODELS / "qwen36-35b-lora"
PORT = 8185


def run(args, **kw):
    print(f"\n=== {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run([str(a) for a in args], cwd=str(ROOT), check=True, **kw)


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    if not ADAPTER.exists():
        sys.exit(f"adapter not found: {ADAPTER} (training still running?)")
    print(f"adapter: {ADAPTER.stat().st_size / 1e6:.1f} MB", flush=True)

    run([PY, "scripts/merge_lora_qwen36.py"])
    run([PY, "c/tools/convert_qwen36.py", "--model", MERGED, "--out", CANDIDATE,
         "--ebits", "4", "--gs", "64", "--no-readme"])

    env = dict(os.environ, COLI_VULKAN="1")
    log = open(RESULTS / "serve_finetune.log", "wb")
    serve = subprocess.Popen(
        [str(PYW), "c/coli", "serve", "--model", str(CANDIDATE),
         "--auto-tier", "--policy", "quality", "--ctx", "8192", "--ngen", "700",
         "--port", str(PORT), "--api-key", "local", "--model-id", "qwen36-35b-lora"],
        cwd=str(ROOT), stdout=log, stderr=log, env=env, creationflags=0x08000000)
    print(f"candidate serve pid {serve.pid} on :{PORT}", flush=True)
    try:
        end = time.time() + 300
        while time.time() < end and not healthy():
            if serve.poll() is not None:
                sys.exit("candidate serve died at startup; see results/serve_finetune.log")
            time.sleep(3)
        if not healthy():
            sys.exit("candidate serve never became healthy")

        for suite in ("hard", "base"):
            run([PY, "scripts/bench_best_model.py",
                 "--base", f"http://127.0.0.1:{PORT}/v1",
                 "--model", "qwen36-35b-lora", "--label", f"finetune_{suite}",
                 "--suite", suite, "--max-tokens", "700", "--no-think"])
    finally:
        log.close()
        subprocess.run(["taskkill", "/PID", str(serve.pid), "/T", "/F"],
                       capture_output=True)
        print("candidate serve stopped", flush=True)

    run([PY, "scripts/make_finetune_graph.py"])

    def s(name):
        return json.loads((RESULTS / name).read_text())["summary"]
    bh, bb = s("bench_best_qwen36_hard.json"), s("bench_best_qwen36_base.json")
    ch, cb = s("bench_best_finetune_hard.json"), s("bench_best_finetune_base.json")
    base_total = bh["passed"] + bb["passed"]
    cand_total = ch["passed"] + cb["passed"]
    verdict = {
        "original": {"hard": f"{bh['passed']}/{bh['total']}", "base": f"{bb['passed']}/{bb['total']}",
                     "tok_s_hard": bh["tokens_per_s"]},
        "candidate": {"hard": f"{ch['passed']}/{ch['total']}", "base": f"{cb['passed']}/{cb['total']}",
                      "tok_s_hard": ch["tokens_per_s"]},
        "regression": cand_total < base_total,
        "recommendation": "keep original" if cand_total < base_total else "promote candidate (no regression)",
        "note": "hard suite overlaps SFT training data (retention); base suite mostly held out (generalization)",
    }
    (RESULTS / "finetune_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()