#!/usr/bin/env python3
"""Benchmark the Fennec engine against AirLLM alone on the same machine.

    python bench_engine_vs_airllm.py --backend both --tokens 32

* engine  - starts `fennec serve` on a free loopback port and streams the fixed
            prompts through /v1/chat/completions, recording time-to-first-token
            and decode tokens/second. Fennec's native binary self-launches; a
            Colibri checkout (`python c/coli serve`) is the fallback, resolved by
            engine_paths exactly as fennec-core does.
* airllm  - loads a model through AirLLM's layer streaming (no engine code) and
            generates the same prompts greedily. AirLLM returns the sequence in
            one call, so its end-to-end figure includes prefill; the timed 1-token
            warm-up is subtracted to also report a decode-only rate. Checkpoints
            whose decoder is nested (model.language_model.layers) get explicit
            layer names, which stock AirLLM cannot infer.

Both phases use the same prompts. AirLLM streams every layer from disk per token,
so --airllm-tokens can shrink its budget independently of --tokens. The engine
process this script starts is always stopped; nothing else is touched. Results are
written to results/engine_vs_airllm.{json,md}.
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
import threading
import time
import urllib.error
import urllib.request
from importlib import import_module
from pathlib import Path

from engine_paths import (EngineError, default_engine_root, engine_env, env_alias,
                          resolve, serve_command)

PROMPTS = [
    "Reply with exactly the single word: pong",
    "Name the capital of France in one short sentence.",
    "Write a Python one-liner that reverses a list.",
]
RESULTS = Path(__file__).resolve().parents[1] / "results"
# Neither Qwen3.6 container ships a chat template; this is the Qwen im_start format.
CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' "
    "+ message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def nested_layer_names(model_dir):
    """AirLLM assumes a `model.layers` trunk. Checkpoints that nest the decoder under
    `model.language_model` make it parse 'layers' as an integer and refuse to load, so
    name the trunk explicitly. Returns None for the layout AirLLM already infers."""
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return None
    keys = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    for trunk in sorted({key.split(".layers.")[0] for key in keys if ".layers." in key}):
        if trunk == "model":
            return None
        if f"{trunk}.embed_tokens.weight" in keys and f"{trunk}.norm.weight" in keys:
            return {"embed": f"{trunk}.embed_tokens", "layer_prefix": f"{trunk}.layers",
                    "norm": f"{trunk}.norm", "lm_head": "lm_head", "expert_prefix": "mlp.experts"}
    return None


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
                       "stream": True, "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {api_key}"})
    started = time.monotonic()
    first_token = None
    pieces = []
    completion_tokens = None
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
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                completion_tokens = int(usage["completion_tokens"])
            for choice in chunk.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    if first_token is None:
                        first_token = time.monotonic()
                    pieces.append(text)
    finished = time.monotonic()
    return {"text": "".join(pieces), "ttft_s": round((first_token or finished) - started, 3),
            "seconds": round(finished - started, 3), "completion_tokens": completion_tokens}


class PeakMemory:
    """Sample resident RAM (a process tree, or this process) and GPU VRAM in the background."""

    def __init__(self, pid=None, interval=0.5):
        self.interval = interval
        self.pid = pid
        self.peak_rss_gib = None
        self.peak_vram_gib = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        try:
            import psutil
        except ImportError:
            psutil = None
        if psutil is not None:
            try:
                root = psutil.Process(self.pid) if self.pid else psutil.Process()
                procs = [root, *root.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in procs if p.is_running())
                self.peak_rss_gib = round(max(self.peak_rss_gib or 0, rss / (1024 ** 3)), 2)
            except Exception:  # noqa: BLE001 - a vanished child must not abort the benchmark
                pass
        query = shutil.which("nvidia-smi")
        if query:
            try:
                out = subprocess.run([query, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10)
                used = max(int(line.strip()) for line in out.stdout.splitlines() if line.strip().isdigit())
                self.peak_vram_gib = round(max(self.peak_vram_gib or 0, used / 1024), 2)
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self._sample()
                self._stop.wait(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return False

    def report(self):
        return {"peak_rss_gib": self.peak_rss_gib, "peak_vram_gib": self.peak_vram_gib}


def bench_engine(args):
    try:
        engine = resolve(args.engine_root)
    except EngineError as exc:
        return {"status": "error", "error": str(exc)}
    port = free_port(args.port)
    base_url = f"http://127.0.0.1:{port}"
    log_path = RESULTS / "engine_vs_airllm_engine.log"
    # Neither spelling may leak a GPU request into the cpu/default arms.
    dropped = {"COLI_CUDA", "COLI_METAL", "COLI_VK_DEV2",
               "FENNEC_CUDA", "FENNEC_METAL", "FENNEC_VK_DEV2"}
    env = engine_env({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                     base={k: v for k, v in os.environ.items() if k not in dropped})
    if args.backend_engine == "vulkan":
        env["COLI_VULKAN"] = "1"
    if args.engine_api_key:
        # Fennec reads the key from the environment; Colibri also takes --api-key.
        env["COLI_API_KEY"] = args.engine_api_key
    command = serve_command(engine, Path(args.model).resolve(), port,
                            extra=["--policy", "quality", "--ctx", "4096",
                                   "--ngen", str(args.tokens + 16)],
                            api_key=args.engine_api_key,
                            python=args.engine_python or str(Path(sys.executable)))
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=str(engine.root), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env)
    report = {"status": "ok", "command": command, "log": str(log_path),
              "engine": engine.kind, "backend": args.backend_engine, "port": port}
    try:
        model_id = wait_ready(base_url, process, args.startup_timeout, args.engine_api_key or "local")
        report["startup_s"] = round(time.monotonic() - started, 2)
        report["model_id"] = model_id
        runs = []
        with PeakMemory(pid=process.pid) as memory:
            for prompt in args.prompt_list:
                stream_chat(base_url, args.engine_api_key or "local", model_id, prompt, 1)  # warm-up
                run = stream_chat(base_url, args.engine_api_key or "local", model_id, prompt, args.tokens)
                tokens = run["completion_tokens"] or len(run["text"].split())
                decode = max(run["seconds"] - run["ttft_s"], 1e-6)
                run.update(prompt=prompt, tokens=tokens,
                           counted="usage" if run["completion_tokens"] else "whitespace-approx",
                           tokens_per_s=round(max(tokens - 1, 1) / decode, 3))
                runs.append(run)
        report.update(memory.report())
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
    budget = args.airllm_tokens or args.tokens
    report = {"status": "ok", "model": str(Path(args.airllm_model).resolve()),
              "max_new_tokens": budget}
    try:
        import torch
        from transformers import AutoTokenizer
        try:
            from air_llm import AutoModel
            package = "air_llm"
        except ImportError:
            from airllm import AutoModel
            package = "airllm"
    except Exception as exc:  # noqa: BLE001
        report.update(status="error", error=f"import failed: {type(exc).__name__}: {exc}")
        return report
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    report["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
    report.update(package=package, device=device)
    model_dir = str(Path(args.airllm_model).resolve())
    names = nested_layer_names(model_dir)
    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if names:
            base = import_module(f"{package}.airllm_base").AirLLMBaseModel

            class NestedTrunk(base):
                def set_layer_names_dict(self):
                    self.layer_names_dict = names

            factory = NestedTrunk
            report["adapter"] = f"explicit layer names for {names['layer_prefix']}"
        else:
            factory = AutoModel.from_pretrained
        model = factory(
            model_dir, device=device, max_seq_len=args.airllm_seq_len,
            layer_shards_saving_path=str(Path(args.airllm_shards).resolve()) if args.airllm_shards else None,
            compression=None, profiling_mode=False)
    except Exception as exc:  # noqa: BLE001
        report.update(status="error", error=f"load failed: {type(exc).__name__}: {exc}",
                      load_s=round(time.monotonic() - started, 2))
        return report
    report["load_s"] = round(time.monotonic() - started, 2)
    report["chat_template"] = "from the model" if tokenizer.chat_template else "explicit Qwen fallback"
    runs = []
    with PeakMemory() as memory:
        for prompt in args.prompt_list:
            messages = [{"role": "user", "content": prompt}]
            extra = {} if tokenizer.chat_template else {"chat_template": CHAT_TEMPLATE}
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.airllm_seq_len)
            warm_began = time.monotonic()
            with torch.no_grad():  # warm-up, mirroring the engine phase
                model.generate(inputs["input_ids"], max_new_tokens=1, use_cache=True, do_sample=False)
            warm_s = time.monotonic() - warm_began
            began = time.monotonic()
            try:
                with torch.no_grad():
                    generated = model.generate(inputs["input_ids"], max_new_tokens=budget,
                                               use_cache=True, do_sample=False)
            except Exception as exc:  # noqa: BLE001
                report.update(status="error", error=f"generate failed: {type(exc).__name__}: {exc}",
                              prompt=prompt, runs=runs)
                return report
            elapsed = time.monotonic() - began
            produced = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
            output = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            # warm-up = prefill + 1 decode, run = prefill + N decode: subtract to isolate decode.
            decode_s = (elapsed - warm_s) / max(produced - 1, 1)
            runs.append({"prompt": prompt, "seconds": round(elapsed, 3), "tokens": produced,
                         "tokens_per_s": round(produced / elapsed, 3) if elapsed > 0 else None,
                         "warmup_s": round(warm_s, 3),
                         "decode_tokens_per_s": round(1 / decode_s, 4) if decode_s > 0 else None,
                         "text": output})
    report.update(memory.report())
    report["runs"] = runs
    values = sorted(run["tokens_per_s"] for run in runs)
    report["median_tokens_per_s"] = values[len(values) // 2]
    decoded = sorted(run["decode_tokens_per_s"] for run in runs if run["decode_tokens_per_s"])
    if decoded:
        report["median_decode_tokens_per_s"] = decoded[len(decoded) // 2]
    return report


def engine_label(engine):
    """Name the engine that actually ran; archived reports predate the Fennec port."""
    return "Fennec engine" if (engine or {}).get("engine") == "fennec" else "Colibri engine"


def verdict(engine, airllm):
    label = engine_label(engine)
    if engine.get("status") != "ok":
        return None
    if airllm.get("status") != "ok":
        return (f"keep the {label} - AirLLM alone never completed the run "
                f"({airllm.get('error', 'not run')}).")
    rate = engine.get("median_tokens_per_s")
    other = airllm.get("median_decode_tokens_per_s") or airllm.get("median_tokens_per_s")
    if not rate or not other:
        return None
    if other > rate:
        return f"AirLLM alone is faster ({other} vs {rate} tok/s); dropping the engine is worth considering."
    return (f"keep the {label} - {rate} tok/s decode against {other} tok/s for AirLLM alone "
            f"({rate / other:.0f}x), both measured decode-only.")


def markdown(report):
    engine, airllm = report.get("engine") or {}, report.get("airllm") or {}
    label = engine_label(engine)
    lines = [f"# {label} vs AirLLM alone", "",
             f"Generated {report['generated_at']} on {report['host']['platform']} "
             f"({report['host']['cpu_count']} threads, Python {report['host']['python']}).", "",
             f"- Engine model: `{report['model']['engine']}`",
             f"- AirLLM model: `{report['model']['airllm']}`",
             f"- Prompts: {len(report['prompts'])}, {report['tokens']} new tokens each, greedy/stream"]
    if airllm.get("max_new_tokens") not in (None, report["tokens"]):
        lines.append(f"- AirLLM budget: {airllm['max_new_tokens']} new tokens (it streams every layer "
                     "per token, so a full sweep can take hours)")
    lines += ["", "| Backend | Status | Load s | Median TTFT s | Decode tok/s | End-to-end tok/s | Peak RAM GiB | Peak VRAM GiB |",
              "|---|---|---|---|---|---|---|---|"]
    for name, data in ((label, engine), ("AirLLM only", airllm)):
        end_to_end = data.get("median_tokens_per_s", "n/a")
        decode = data.get("median_decode_tokens_per_s", "n/a")
        if name == label:
            decode, end_to_end = end_to_end, "n/a"
            rates = sorted(r["tokens"] / r["seconds"] for r in data.get("runs") or [] if r.get("seconds"))
            if rates:
                end_to_end = round(rates[len(rates) // 2], 3)
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, data.get("status", "not run"), data.get("load_s", data.get("startup_s", "n/a")),
            data.get("median_ttft_s", "n/a"), decode, end_to_end,
            data.get("peak_rss_gib", "n/a"), data.get("peak_vram_gib", "n/a")))
    lines += ["", "Counting: the engine reports real `usage.completion_tokens`; its decode rate excludes",
              "time-to-first-token. AirLLM returns the whole sequence in one call, so its end-to-end rate",
              "includes prefill and it has no separate TTFT. To compare like with like, AirLLM's decode",
              "rate subtracts the timed 1-token warm-up (prefill + 1 decode) from the timed run (prefill +",
              "N decode) and divides by N-1. Both phases run that warm-up first. Peak RAM is the sampled",
              "resident set of the whole process tree.", ""]
    for name, data in ((label, engine), ("AirLLM only", airllm)):
        if data.get("error"):
            lines.append(f"- **{name} error:** `{data['error']}`")
        if data.get("torch"):
            lines.append(f"- {name}: {data.get('package', 'air_llm')} on `{data.get('device', '?')}`, "
                         f"torch {data['torch']['version']}, cuda={data['torch']['cuda']}")
        if data.get("backend"):
            lines.append(f"- {name} backend: `{data['backend']}`")
        if data.get("adapter"):
            lines.append(f"- {name} adapter: {data['adapter']}")
        if data.get("chat_template"):
            lines.append(f"- {name} chat template: {data['chat_template']}")
    faster = verdict(engine, airllm)
    if faster:
        lines += ["", f"**Verdict:** {faster}"]
    lines += ["", "## Per-prompt detail", ""]
    for name, data in ((label, engine), ("AirLLM only", airllm)):
        lines.append(f"### {name}")
        for run in data.get("runs") or []:
            text = (run.get("text") or "").strip().replace("\n", " ")[:120]
            rate = run.get("decode_tokens_per_s") or run["tokens_per_s"]
            lines.append(f"- `{rate}` tok/s decode, {run['seconds']}s for {run['tokens']} tok, "
                         f"ttft {run.get('ttft_s', 'n/a')}s - {text}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("engine", "airllm", "both"), default="both")
    ap.add_argument("--backend-engine", choices=("vulkan", "cpu", "default"), default="vulkan",
                    help="COLI_VULKAN=1 / FENNEC_VULKAN=1 (the shipped GPU path), CPU only, or engine default")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS),
                    help=f"how many of the {len(PROMPTS)} fixed prompts to run")
    ap.add_argument("--engine-root", default=default_engine_root(),
                    help="Fennec checkout (target/release/fennec) or a Colibri checkout with c/coli")
    ap.add_argument("--engine-python", default=None,
                    help="interpreter for the Colibri launcher; ignored by the native Fennec binary")
    ap.add_argument("--engine-api-key", default="local")
    ap.add_argument("--model", default=os.environ.get("FENNEC_ENGINE_MODEL") or env_alias("MODEL") or
                    str(Path(os.environ.get("FENNEC_MODELS") or Path.home() / "fennec-models") / "qwen36-35b"))
    ap.add_argument("--airllm-model", default=os.environ.get("FENNEC_AIRLLM_MODEL") or
                    str(Path(os.environ.get("FENNEC_MODELS") or Path.home() / "fennec-models") / "Qwen3.6-35B-A3B-bf16"))
    ap.add_argument("--airllm-shards", default=os.environ.get("FENNEC_AIRLLM_SHARDS") or
                    str(Path(os.environ.get("FENNEC_CACHE") or Path.home() / ".cache" / "fennec") / "airllm-shards"))
    ap.add_argument("--airllm-seq-len", type=int, default=512)
    ap.add_argument("--airllm-tokens", type=int, default=None,
                    help="separate AirLLM token budget; it streams every layer per token, "
                         "so a full sweep can take hours (default: --tokens)")
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
