#!/usr/bin/env python3
"""Benchmark the Colibri engine against AirLLM alone on the same machine.

    python bench_engine_vs_airllm.py --backend both --tokens 32

* engine  - starts `c/coli serve` on a free loopback port and streams the fixed
            prompts through /v1/chat/completions, recording time-to-first-token
            and decode tokens/second.
* airllm  - loads a model through AirLLM's layer streaming (no engine code) and
            generates the same prompts greedily. AirLLM returns the sequence in
            one call, so its figure is end-to-end tokens/second including
            prefill.

Both phases use the same prompts and token budget. The engine process this
script starts is always stopped; nothing else is touched. Results are written to
results/engine_vs_airllm.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    "Reply with exactly the single word: pong",
    "Name the capital of France in one short sentence.",
    "Write a Python one-liner that reverses a list.",
]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def free_port(preferred=8197):
    for port in (preferred, preferred + 1, preferred + 2, preferred + 3):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise SystemExit("no free loopback port near 8197")


def host_report():
    total = shutil.disk_usage(Path.home()).total / (1024 ** 3)
    return {"platform": platform.platform(), "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0], "home_volume_gib": round(total, 1)}


def wait_ready(base_url, process, timeout, api_key="local"):
    request = urllib.request.Request(base_url + "/v1/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"engine exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8", "replace"))
                    items = payload.get("data") if isinstance(payload, dict) else None
                    if not items or not items[0].get("id"):
                        raise RuntimeError(f"/v1/models did not name the served model: {payload!r}")
                    return items[0]["id"]
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    raise RuntimeError(f"engine did not answer /v1/models within {timeout}s")


def stream_chat(base_url, api_key, model_id, prompt, max_tokens):
    body = json.dumps({"model": model_id, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {api_key}"})
    started = time.monotonic()
    first_token = None
    pieces = []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            for choice in chunk.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    if first_token is None:
                        first_token = time.monotonic()
                    pieces.append(text)
    finished = time.monotonic()
    return {"text": "".join(pieces), "ttft_s": round((first_token or finished) - started, 3),
            "seconds": round(finished - started, 3)}


def bench_engine(args):
    engine_root = Path(args.engine_root)
    launcher = engine_root / "c" / "coli"
    if not launcher.is_file():
        return {"status": "error", "error": f"engine launcher missing: {launcher}"}
    python = args.engine_python or str(Path(sys.executable))
    port = free_port(args.port)
    base_url = f"http://127.0.0.1:{port}"
    log_path = RESULTS / "engine_vs_airllm_engine.log"
    env = {k: v for k, v in os.environ.items()
           if k not in {"COLI_CUDA", "COLI_METAL", "COLI_VK_DEV2"}}
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if args.backend_engine == "vulkan":
        env["COLI_VULKAN"] = "1"
    command = [python, str(launcher), "serve", "--model", str(Path(args.model).resolve()),
               "--auto-tier", "--policy", "quality", "--ctx", "4096",
               "--ngen", str(args.tokens + 16), "--port", str(port)]
    if args.engine_api_key:
        command += ["--api-key", args.engine_api_key]
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=str(engine_root), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env)
    report = {"status": "ok", "command": command, "log": str(log_path),
              "backend": args.backend_engine, "port": port}
    try:
        model_id = wait_ready(base_url, process, args.startup_timeout, args.engine_api_key or "local")
        report["startup_s"] = round(time.monotonic() - started, 2)
        report["model_id"] = model_id
        runs = []
        for prompt in args.prompt_list:
            stream_chat(base_url, args.engine_api_key or "local", model_id, prompt, 1)  # warm-up
            run = stream_chat(base_url, args.engine_api_key or "local", model_id, prompt, args.tokens)
            tokens = len(run["text"].split())
            decode = max(run["seconds"] - run["ttft_s"], 1e-6)
            run.update(prompt=prompt, approx_tokens=tokens,
                       tokens_per_s=round(max(tokens - 1, 1) / decode, 3))
            runs.append(run)
        report["runs"] = runs
        values = sorted(run["tokens_per_s"] for run in runs)
        report["median_tokens_per_s"] = values[len(values) // 2]
        report["median_ttft_s"] = sorted(run["ttft_s"] for run in runs)[len(runs) // 2]
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    return report


def bench_airllm(args):
    report = {"status": "ok", "model": str(Path(args.airllm_model).resolve()),
              "max_new_tokens": args.tokens}
    try:
        import torch
        from transformers import AutoTokenizer
        from air_llm import AutoModel
    except Exception as exc:  # noqa: BLE001
        report.update(status="error", error=f"import failed: {type(exc).__name__}: {exc}")
        return report
    report["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(Path(args.airllm_model).resolve()))
        model = AutoModel.from_pretrained(
            str(Path(args.airllm_model).resolve()), max_seq_len=args.airllm_seq_len,
            layer_shards_saving_path=str(Path(args.airllm_shards).resolve()) if args.airllm_shards else None,
            compression=None, profiling_mode=False)
    except Exception as exc:  # noqa: BLE001
        report.update(status="error", error=f"load failed: {type(exc).__name__}: {exc}",
                      load_s=round(time.monotonic() - started, 2))
        return report
    report["load_s"] = round(time.monotonic() - started, 2)
    runs = []
    for prompt in args.prompt_list:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.airllm_seq_len)
        began = time.monotonic()
        try:
            with torch.no_grad():
                generated = model.generate(inputs["input_ids"], max_new_tokens=args.tokens,
                                           use_cache=True, do_sample=False)
        except Exception as exc:  # noqa: BLE001
            report.update(status="error", error=f"generate failed: {type(exc).__name__}: {exc}",
                          prompt=prompt, runs=runs)
            return report
        elapsed = time.monotonic() - began
        produced = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
        output = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        runs.append({"prompt": prompt, "seconds": round(elapsed, 3), "tokens": produced,
                     "tokens_per_s": round(produced / elapsed, 3) if elapsed > 0 else None,
                     "text": output})
    report["runs"] = runs
    values = sorted(run["tokens_per_s"] for run in runs)
    report["median_tokens_per_s"] = values[len(values) // 2]
    return report


def markdown(report):
    engine, airllm = report.get("engine") or {}, report.get("airllm") or {}
    lines = ["# Colibri engine vs AirLLM alone", "",
             f"Generated {report['generated_at']} on {report['host']['platform']} "
             f"({report['host']['cpu_count']} threads, Python {report['host']['python']}).", "",
             f"- Engine model: `{report['model']['engine']}`",
             f"- AirLLM model: `{report['model']['airllm']}`",
             f"- Prompts: {len(report['prompts'])}, {report['tokens']} new tokens each, greedy/stream", "",
             "| Backend | Status | Median TTFT | Median tokens/s | Notes |", "|---|---|---|---|---|"]
    for name, data in (("Colibri engine", engine), ("AirLLM only", airllm)):
        status = data.get("status", "not run")
        ttft = data.get("median_ttft_s", "n/a")
        rate = data.get("median_tokens_per_s", "n/a")
        note = data.get("error") or (f"torch {data['torch']['version']}, cuda={data['torch']['cuda']}, "
                                     f"load {data.get('load_s', 'n/a')}s") if data.get("torch") else data.get("backend", "")
        lines.append(f"| {name} | {status} | {ttft} | {rate} | {note} |")
    lines += ["", "## Per-prompt detail", ""]
    for name, data in (("Colibri engine", engine), ("AirLLM only", airllm)):
        lines.append(f"### {name}")
        for run in data.get("runs") or []:
            text = (run.get("text") or "").strip().replace("\n", " ")[:120]
            lines.append(f"- `{run['tokens_per_s']}` tok/s, {run['seconds']}s, "
                         f"ttft {run.get('ttft_s', 'n/a')}s - {text}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("engine", "airllm", "both"), default="both")
    ap.add_argument("--backend-engine", choices=("vulkan", "cpu", "default"), default="vulkan",
                    help="COLI_VULKAN=1 (the shipped GPU path), CPU only, or engine default")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS),
                    help=f"how many of the {len(PROMPTS)} fixed prompts to run")
    ap.add_argument("--engine-root", default=os.environ.get("FENNEC_HOME") or str(Path.home() / "Colibri"))
    ap.add_argument("--engine-python", default=None)
    ap.add_argument("--engine-api-key", default="local")
    ap.add_argument("--model", default=os.environ.get("FENNEC_ENGINE_MODEL") or
                    str(Path(os.environ.get("FENNEC_MODELS") or Path.home() / "fennec-models") / "qwen36-35b"))
    ap.add_argument("--airllm-model", default=os.environ.get("FENNEC_AIRLLM_MODEL") or
                    str(Path(os.environ.get("FENNEC_MODELS") or Path.home() / "fennec-models") / "Qwen3.6-35B-A3B-bf16"))
    ap.add_argument("--airllm-shards", default=os.environ.get("FENNEC_AIRLLM_SHARDS") or
                    str(Path(os.environ.get("FENNEC_CACHE") or Path.home() / ".cache" / "fennec") / "airllm-shards"))
    ap.add_argument("--airllm-seq-len", type=int, default=512)
    ap.add_argument("--port", type=int, default=8197)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args.prompt_list = PROMPTS[:max(1, min(args.prompts, len(PROMPTS)))]
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "host": host_report(),
              "prompts": args.prompt_list, "tokens": args.tokens,
              "model": {"engine": args.model, "airllm": args.airllm_model}}
    if args.backend in ("engine", "both"):
        print("benchmarking the engine...", flush=True)
        report["engine"] = bench_engine(args)
        print(json.dumps({k: v for k, v in report["engine"].items() if k != "runs"}, indent=2), flush=True)
    if args.backend in ("airllm", "both"):
        print("benchmarking AirLLM...", flush=True)
        report["airllm"] = bench_airllm(args)
        print(json.dumps({k: v for k, v in report["airllm"].items() if k != "runs"}, indent=2), flush=True)
    json_path = out / "engine_vs_airllm.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "engine_vs_airllm.md").write_text(markdown(report), encoding="utf-8")
    print(f"wrote {json_path} and engine_vs_airllm.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
