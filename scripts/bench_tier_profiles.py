#!/usr/bin/env python3
"""Find the fastest tier layout for this machine, measured rather than assumed.

Each profile starts the engine on the same model with the same prompt and token
budget, then records the streaming time-to-first-token and decode rate. The
engine is spawned and stopped by this script; nothing else is touched.

    python bench_tier_profiles.py --pairs 1

Profiles are the knobs `fennec serve` exposes for placement: the resource policy,
the VRAM budget, the host-RAM expert pool (COLI_VK_TIER_HOST_GB, alias
FENNEC_VK_TIER_HOST_GB) and the CPU thread count. NPU is absent on purpose:
`--xdna` is AMD-only and the Intel NPU has no path into the engine.

The engine is resolved by engine_paths: Fennec's native binary self-launches,
and a Colibri checkout (`python c/coli serve`) is the fallback. `--auto-tier` is
always applied, so the profiles below only carry what differs between them.

Results: results/tier_profiles.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from engine_paths import (EngineError, default_engine_root, env_alias, resolve,
                          serve_command, strip_engine_vars)

RESULTS = Path(__file__).resolve().parents[1] / "results"
PROMPT = "Write the numbers from 1 to 50 separated by commas, nothing else."

PROFILES = [
    ("pool 4 GB, gpu auto", {"COLI_VULKAN": "1", "COLI_VK_TIER_HOST_GB": "4"},
     ["--policy", "quality"]),
    ("pool 4 GB + gpu 4 GB", {"COLI_VULKAN": "1", "COLI_VK_TIER_HOST_GB": "4",
                              "COLI_VK_TIER_GB": "4"},
     ["--policy", "quality"]),
    ("pool 6 GB + gpu 4 GB", {"COLI_VULKAN": "1", "COLI_VK_TIER_HOST_GB": "6",
                              "COLI_VK_TIER_GB": "4"},
     ["--policy", "quality"]),
]


def free_port(start=8196):
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise SystemExit("no free loopback port")


def wait_ready(base, process, timeout=240):
    request = urllib.request.Request(base + "/v1/models", headers={"Authorization": "Bearer local"})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"engine exited with {process.returncode}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            return payload["data"][0]["id"]
        except (urllib.error.URLError, OSError, KeyError, IndexError):
            time.sleep(1.0)
    raise RuntimeError(f"engine not ready within {timeout}s")


def stream(base, model, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    request = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer local"})
    started = time.monotonic()
    first = None
    tokens = 0
    usage = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                payload = json.loads(chunk)
            except ValueError:
                continue
            usage = payload.get("usage") or usage
            for choice in payload.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if first is None:
                        first = time.monotonic()
                    tokens += 1
    finished = time.monotonic()
    ttft = (first or finished) - started
    decode = max(finished - (first or finished), 1e-6)
    return {"ttft_s": round(ttft, 2), "total_s": round(finished - started, 2),
            "tokens": tokens, "tokens_per_s": round(max(tokens - 1, 1) / decode, 2),
            "usage": usage}


def stop_tree(process):
    """Kill the engine tree (Colibri's launcher plus its child); on Windows terminate() orphans it."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        process.terminate()
    try:
        process.wait(timeout=40)
    except subprocess.TimeoutExpired:
        process.kill()


def run_profile(name, extra_env, extra_args, args, engine):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = RESULTS / f"tier-{name.replace(' ', '-').replace('+', '')}.log"
    env = strip_engine_vars(os.environ)
    env.update(extra_env, COLI_API_KEY="local", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    command = serve_command(engine, args.model, port,
                            extra=["--ctx", str(args.ctx), "--ngen", str(args.ngen), *extra_args],
                            api_key="local")
    report = {"profile": name, "engine": engine.kind, "args": extra_args,
              "env": extra_env, "log": str(log)}
    started = time.monotonic()
    with log.open("wb") as handle:
        process = subprocess.Popen(command, cwd=str(engine.root), env=env,
                                   stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    try:
        model = wait_ready(base, process)
        report["startup_s"] = round(time.monotonic() - started, 1)
        stream(base, model, 4)  # warm-up, discarded
        runs = [stream(base, model, args.tokens) for _ in range(args.pairs)]
        report["runs"] = runs
        report["median_tokens_per_s"] = sorted(r["tokens_per_s"] for r in runs)[len(runs) // 2]
        report["median_ttft_s"] = sorted(r["ttft_s"] for r in runs)[len(runs) // 2]
        report["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        stop_tree(process)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine-root", default=default_engine_root(),
                        help="Fennec checkout (target/release/fennec) or a Colibri checkout with c/coli")
    parser.add_argument("--model", default=env_alias("MODEL") or "",
                        help="model directory; FENNEC_MODEL or COLI_MODEL")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--ngen", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--only", default=None, help="substring of one profile name")
    parser.add_argument("--out", default=str(RESULTS / "tier_profiles"))
    args = parser.parse_args(argv)
    if not args.model:
        raise SystemExit("--model or FENNEC_MODEL (or COLI_MODEL) is required")
    try:
        engine = resolve(args.engine_root)
    except EngineError as exc:
        raise SystemExit(str(exc))

    reports = []
    for name, env, extra in PROFILES:
        if args.only and args.only.lower() not in name.lower():
            continue
        print(f"--- {name} ---", flush=True)
        report = run_profile(name, env, extra, args, engine)
        reports.append(report)
        if report["status"] == "ok":
            print(f"    {report['median_tokens_per_s']} tok/s, ttft {report['median_ttft_s']} s, "
                  f"start {report['startup_s']} s", flush=True)
        else:
            print(f"    FAILED: {report['error']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"model": args.model, "ctx": args.ctx,
                                                    "tokens": args.tokens, "prompt": PROMPT,
                                                    "engine": engine.kind,
                                                    "engine_binary": str(engine.binary),
                                                    "reports": reports}, indent=2) + "\n",
                                        encoding="utf-8")
    out.with_suffix(".md").write_text(render(reports, args), encoding="utf-8")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


def render(reports, args):
    lines = ["# Engine tier profiles, same prompt and token budget", "",
             f"`{args.model}`, `--ctx {args.ctx}`, {args.tokens} tokens, greedy. "
             "Higher is better; TTFT is prefill.", "",
             "| Profile | Status | median tok/s | median TTFT | startup |",
             "|---|---|---|---|---|"]
    for report in reports:
        if report["status"] == "ok":
            lines.append(f"| {report['profile']} | ok | **{report['median_tokens_per_s']}** | "
                         f"{report['median_ttft_s']} s | {report['startup_s']} s |")
        else:
            lines.append(f"| {report['profile']} | error | — | — | — |")
    best = max((r for r in reports if r["status"] == "ok"),
               key=lambda r: r["median_tokens_per_s"], default=None)
    if best:
        lines += ["", f"Fastest: **{best['profile']}** at {best['median_tokens_per_s']} tok/s "
                      f"({best['median_ttft_s']} s TTFT)."]
    lines += ["", "NPU is not a candidate: `--xdna` is AMD-only and the Intel NPU has no "
                  "path into this engine.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
