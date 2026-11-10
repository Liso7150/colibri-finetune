#!/usr/bin/env python3
"""Does the compacted agent payload actually cost less on the live engine?

One fixed agent transcript, sent twice through /v1/chat/completions:

* ``old`` - what the loop sent before fennec-core 1.4.1: every tool result
            sliced to 6,000 characters, duplicates repeated verbatim.
* ``new`` - ``fennec_core.context.compact()`` with the shipped defaults.

Both arms keep the same system prompt, task, temperature 0 and max_tokens, and
the arms alternate each pair so warm-up and prefix-cache drift land on both.
Reported: exact ``prompt_tokens`` from the engine's usage, median
time-to-first-token (prefill) and median total seconds, plus every trial.

    python bench_context_payload.py --pairs 5 --tokens 16 --port 8197

Results are written to results/context_payload.{json,md}. The engine must
already be serving; this script never starts or stops it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _core_root(explicit):
    root = Path(explicit) if explicit else Path(__file__).resolve().parents[2] / "fennec-core"
    if (root / "fennec_core").is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


SYSTEM = ("You are Fennec Desktop, a local coding agent. You edit files in the user's "
          "workspace with the tools provided, you never invent file contents, and you "
          "answer in short, concrete sentences.\n\n" +
          "Workspace rules: paths are relative to the workspace root; read before you "
          "write; prefer small diffs; run the tests you touched.\n" * 6)

TASK = "The suite's test file src/app.py is failing. Read it, find the failing test, and tell me the one-line fix."


def raw_results():
    """One realistic tool round: a big read, a grep, the same read again, a test run."""
    source = "".join(
        f"  {line}\t{'def test_' + f'case_{line}():' if line % 40 == 0 else 'x = compute(value)'}\n"
        for line in range(1, 401))
    read = {"ok": True, "path": "src/app.py", "lines": 400, "truncated": False, "content": source}
    hits = {"ok": True, "count": 150, "truncated": False,
            "hits": [{"file": "src/app.py", "line": n, "text": "    assert handler(payload) == expected"}
                     for n in range(1, 151)]}
    shell = {"ok": True, "exit": 1, "truncated": False,
             "stdout": "".join(f"tests/test_app.py::test_case_{n} PASSED\n" for n in range(1, 401)),
             "stderr": "1 failed, 399 passed in 4.12s\n"}
    return read, hits, shell


def transcript():
    """The message list the loop holds in memory: full, uncompacted tool results."""
    read, hits, shell = raw_results()
    call = lambda cid, name: {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "read_file")]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(read, ensure_ascii=False)},
        {"role": "assistant", "content": "", "tool_calls": [call("c2", "grep")]},
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps(hits, ensure_ascii=False)},
        {"role": "assistant", "content": "", "tool_calls": [call("c3", "read_file")]},
        {"role": "tool", "tool_call_id": "c3", "content": json.dumps(read, ensure_ascii=False)},
        {"role": "assistant", "content": "", "tool_calls": [call("c4", "run_shell")]},
        {"role": "tool", "tool_call_id": "c4", "content": json.dumps(shell, ensure_ascii=False)},
    ]


def old_messages():
    """Pre-1.4.1 loop: hard slice, no dedupe, no budget."""
    return [{**message,
             "content": message["content"][:6000]} if message["role"] == "tool" else message
            for message in transcript()]


def new_messages(context):
    seen = {}
    payload, stats = context.compact(transcript(), seen=seen)
    return payload, stats


def _chars(messages):
    return sum(len(m["content"]) if isinstance(m.get("content"), str) else 0 for m in messages)


def _post(base_url, api_key, body):
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    return urllib.request.urlopen(request, timeout=1800)


def stream_chat(base_url, api_key, model_id, messages, max_tokens):
    body = {"model": model_id, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    started = time.monotonic()
    first_token = None
    pieces = []
    usage = {}
    with _post(base_url, api_key, body) as response:
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
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    if first_token is None:
                        first_token = time.monotonic()
                    pieces.append(text)
    finished = time.monotonic()
    return {"ttft_s": round((first_token or finished) - started, 3),
            "seconds": round(finished - started, 3),
            "chars": len("".join(pieces)), "usage": usage}


def exact_usage(base_url, api_key, model_id, messages, max_tokens):
    body = {"model": model_id, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    with _post(base_url, api_key, body) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    return payload.get("usage") or {}


def model_id(base_url, api_key):
    request = urllib.request.Request(base_url + "/v1/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    return payload["data"][0]["id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8197")
    parser.add_argument("--api-key", default=os.environ.get("FENNEC_API_KEY", "local"))
    parser.add_argument("--core-root", default=None)
    parser.add_argument("--pairs", type=int, default=2, help="alternating old/new pairs")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--exact-usage", action="store_true",
                        help="two extra non-streaming calls for exact prompt tokens (one full prefill each)")
    parser.add_argument("--out", default=str(RESULTS / "context_payload"))
    args = parser.parse_args(argv)

    if args.pairs < 1:
        raise SystemExit("--pairs must be >= 1")
    _core_root(args.core_root)
    from fennec_core import context  # imported after the path fix

    old = old_messages()
    new, stats = new_messages(context)
    arms = {"old": old, "new": new}

    try:
        served = model_id(args.base_url, args.api_key)
    except (urllib.error.URLError, OSError, KeyError, IndexError) as exc:
        raise SystemExit(f"engine not reachable at {args.base_url}: {type(exc).__name__}: {exc}")

    report = {"engine": args.base_url, "model": served, "max_tokens": args.tokens,
              "temperature": 0, "pairs": args.pairs,
              "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                       "python": sys.version.split()[0]},
              "compact_stats": stats,
              "arms": {name: {"chars": _chars(messages), "messages": len(messages)}
                       for name, messages in arms.items()},
              "trials": []}

    for name in ("old", "new"):
        report["arms"][name]["prompt_tokens"] = None
        report["arms"][name]["est_prompt_tokens"] = round(_chars(arms[name]) / 4)
    if args.exact_usage:
        # Exact prompt tokens, at the price of one full non-streaming generation
        # per arm (~1.5 min of prefill each on this engine), so it is opt-in.
        print("measuring exact prompt tokens (2 extra generations)...", flush=True)
        for name in ("old", "new"):
            usage = exact_usage(args.base_url, args.api_key, served, arms[name], args.tokens)
            report["arms"][name]["prompt_tokens"] = usage.get("prompt_tokens")
            report["arms"][name]["completion_tokens"] = usage.get("completion_tokens")
            print(f"  {name}: prompt_tokens={report['arms'][name]['prompt_tokens']}", flush=True)

    for pair in range(args.pairs):
        order = ("old", "new") if pair % 2 == 0 else ("new", "old")
        for name in order:
            run = stream_chat(args.base_url, args.api_key, served, arms[name], args.tokens)
            run.update(pair=pair, order=list(order), arm=name, chars=_chars(arms[name]))
            report["trials"].append(run)
            print(f"pair {pair + 1}/{args.pairs}  {name:3s}  {run['chars']:6d} chars  "
                  f"ttft {run['ttft_s']:6.2f}s  total {run['seconds']:6.2f}s", flush=True)

    summary = {}
    for name in ("old", "new"):
        trials = [t for t in report["trials"] if t["arm"] == name]
        summary[name] = {
            "trials": len(trials),
            "chars": report["arms"][name]["chars"],
            "prompt_tokens": report["arms"][name]["prompt_tokens"],
            "est_prompt_tokens": report["arms"][name]["est_prompt_tokens"],
            "median_ttft_s": round(statistics.median(t["ttft_s"] for t in trials), 3),
            "median_seconds": round(statistics.median(t["seconds"] for t in trials), 3),
            "min_ttft_s": min(t["ttft_s"] for t in trials),
            "max_ttft_s": max(t["ttft_s"] for t in trials),
        }
    summary["delta"] = {
        "chars": summary["old"]["chars"] - summary["new"]["chars"],
        "chars_pct": round(100 * (summary["old"]["chars"] - summary["new"]["chars"])
                           / summary["old"]["chars"], 1),
        "ttft_s": round(summary["old"]["median_ttft_s"] - summary["new"]["median_ttft_s"], 3),
        "ttft_pct": round(100 * (summary["old"]["median_ttft_s"] - summary["new"]["median_ttft_s"])
                          / summary["old"]["median_ttft_s"], 1),
        "seconds_s": round(summary["old"]["median_seconds"] - summary["new"]["median_seconds"], 3),
    }
    if summary["old"]["prompt_tokens"] and summary["new"]["prompt_tokens"]:
        summary["delta"]["prompt_tokens"] = (summary["old"]["prompt_tokens"]
                                             - summary["new"]["prompt_tokens"])
    report["summary"] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    print(json.dumps(summary, indent=2))
    return 0


def render_markdown(report):
    summary = report["summary"]
    delta = summary["delta"]
    tokens = ("{}".format(delta["prompt_tokens"]) if "prompt_tokens" in delta else "n/a")
    lines = [
        "# Compacted agent payload vs the old payload, live engine",
        "",
        f"Engine `{report['model']}` at `{report['engine']}`, greedy, "
        f"`max_tokens={report['max_tokens']}`, {report['pairs']} alternating pairs. "
        "Same transcript, same system prompt and task; only the tool payload differs.",
        "",
        "| Arm | chars | prompt tokens | median TTFT | median total |",
        "|---|---|---|---|---|",
    ]
    for name in ("old", "new"):
        arm = summary[name]
        prompt = arm["prompt_tokens"] if arm["prompt_tokens"] else f"≈{arm['est_prompt_tokens']}"
        lines.append(f"| {name} | {arm['chars']:,} | {prompt} | "
                     f"{arm['median_ttft_s']:.2f} s | {arm['median_seconds']:.2f} s |")
    lines += [
        "",
        f"**Delta:** {delta['chars']:,} fewer characters ({delta['chars_pct']} %), "
        f"{tokens} fewer prompt tokens, "
        f"{delta['ttft_s']:+.2f} s TTFT ({delta['ttft_pct']:+.1f} %), "
        f"{delta['seconds_s']:+.2f} s total per step.",
        "",
        "Positive seconds are time saved by the compacted payload. TTFT is the prefill-bound "
        "figure; total includes the same number of decoded tokens in both arms.",
        "",
        "## Trials",
        "",
        "| pair | order | arm | chars | TTFT (s) | total (s) |",
        "|---|---|---|---|---|---|",
    ]
    for trial in report["trials"]:
        lines.append(f"| {trial['pair'] + 1} | {'→'.join(trial['order'])} | {trial['arm']} | "
                     f"{trial['chars']:,} | {trial['ttft_s']:.2f} | {trial['seconds']:.2f} |")
    lines += [
        "",
        "## Payload shape",
        "",
        f"`{json.dumps(report['compact_stats'])}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
