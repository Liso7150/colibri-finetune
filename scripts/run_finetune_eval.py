"""Explicit merge/convert stages and fresh, sequential retention-only evaluation.

--preflight-only resolves and checks paths without importing torch or loading models.
No promotion is performed; legacy hard/base suites cannot establish generalization.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

SCRIPTS = Path(__file__).resolve().parent
SUITE_ROOT = SCRIPTS.parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-root", default=os.environ.get("FENNEC_ENGINE_ROOT") or
                    os.environ.get("FENNEC_HOME") or str(Path.home() / "Colibri"))
    ap.add_argument("--models-dir", default=os.environ.get("FENNEC_MODELS") or
                    str(Path.home() / "fennec-models"))
    ap.add_argument("--results-dir", default=str(SUITE_ROOT / "results"))
    ap.add_argument("--engine-python", help="explicit runtime for engine tools; otherwise current interpreter")
    ap.add_argument("--base", help="bf16 source; defaults under models-dir")
    ap.add_argument("--adapter")
    ap.add_argument("--merged", help="existing merge input or NEW merge output")
    ap.add_argument("--candidate", help="existing container or NEW conversion output")
    ap.add_argument("--baseline", help="baseline container; defaults to models-dir/qwen36-35b")
    ap.add_argument("--scaling", type=float, help="alpha/r, required for adapters without metadata")
    ap.add_argument("--stages", nargs="+", choices=("merge", "convert", "evaluate", "graph"),
                    default=["evaluate"], help="explicit opt-in; executed in pipeline order")
    ap.add_argument("--port", type=int, default=8185)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.engine_root:
        ap.error("set --engine-root or FENNEC_ENGINE_ROOT")
    if not 1 <= args.port <= 65535:
        ap.error("port must be in [1, 65535]")
    if "graph" in args.stages and "evaluate" not in args.stages:
        ap.error("graph stage requires evaluate; use make_finetune_graph.py for existing results")
    return args


def require_file(path):
    if not path.is_file():
        raise ValueError(f"required file missing: {path}")


def require_model(path, converted=False):
    for name in ("config.json", "tokenizer.json"):
        require_file(path / name)
    require_file(path / ("qwen36_meta.json" if converted else "model.safetensors.index.json"))


def require_new(path):
    if path.exists():
        raise ValueError(f"refusing existing output directory: {path}")


def require_free_port(port):
    # Binding (not HTTP health) also catches non-HTTP listeners and wildcard binds.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"port {port} is occupied/unavailable; refusing model launch") from exc


def bench_code(path, results):
    """Redirect the unmodified legacy harness's RESULTS assignment in memory."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "RESULTS" for t in node.targets)]
    if len(assignments) != 1:
        raise ValueError("unsupported engine benchmark: expected one RESULTS assignment")
    assignments[0].value = ast.Constant(str(results))
    ast.fix_missing_locations(tree)
    return compile(tree, str(path), "exec")


def preflight(args):
    root = Path(args.engine_root).resolve()
    models = Path(args.models_dir).resolve()
    results = Path(args.results_dir).resolve()
    run_id = "eval-" + uuid.uuid4().hex
    plan = {"engine_root": root, "models_dir": models, "results_dir": results,
            "run_dir": results / run_id, "run_id": run_id,
            "python": Path(sys.executable).resolve(),
            "engine_python": Path(args.engine_python or sys.executable).resolve(),
            "merge_script": SCRIPTS / "merge_lora_qwen36.py",
            "graph_script": SCRIPTS / "make_finetune_graph.py",
            "convert_script": root / "c/tools/convert_qwen36.py",
            "bench_script": root / "scripts/bench_best_model.py",
            "launcher": root / "c/coli",
            "binary": root / "c" / ("qwen36.exe" if os.name == "nt" else "qwen36"),
            "base": Path(args.base or models / "Qwen3.6-35B-A3B-bf16").resolve(),
            "adapter": Path(args.adapter or models / "qwen36-lora-adapter.pt").resolve(),
            "merged": Path(args.merged or models / ("Qwen3.6-35B-A3B-bf16-lora" +
                           ("-" + run_id if "merge" in args.stages else ""))).resolve(),
            "candidate": Path(args.candidate or models / ("qwen36-35b-lora" +
                              ("-" + run_id if "convert" in args.stages else ""))).resolve(),
            "baseline": Path(args.baseline or models / "qwen36-35b").resolve(),
            "stages": [s for s in ("merge", "convert", "evaluate", "graph") if s in args.stages],
            "port": args.port, "scaling": args.scaling,
            "scope": "retention_only", "promotion_eligible": False,
            "tensor_validation": "deferred to merge; preflight never imports torch"}
    if not root.is_dir() or not models.is_dir():
        raise ValueError("engine-root and models-dir must be existing directories")
    if results.exists() and not results.is_dir():
        raise ValueError("results-dir is not a directory")
    require_new(plan["run_dir"])
    require_file(plan["python"])
    require_file(plan["engine_python"])
    outputs = [plan[s] for stage, s in (("merge", "merged"), ("convert", "candidate"))
               if stage in args.stages]
    inputs = [plan[k] for k in ("base", "baseline", "adapter")]
    for output in outputs:
        require_new(output)
        for other in inputs + [p for p in outputs if p != output]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError(f"model output conflicts with another path: {output}")
    if "merge" in args.stages and "convert" in args.stages and plan["merged"] == plan["candidate"]:
        raise ValueError("merge and conversion outputs must differ")
    if "merge" in args.stages:
        require_file(plan["merge_script"])
        require_model(plan["base"])
        require_file(plan["adapter"])
        from merge_lora_qwen36 import adapter_metadata
        plan["scaling"], _ = adapter_metadata(plan["adapter"], plan["base"], args.scaling)
    if "convert" in args.stages:
        require_file(plan["convert_script"])
        if "merge" not in args.stages:
            require_model(plan["merged"])
            if plan["candidate"] == plan["merged"] or plan["merged"] in plan["candidate"].parents:
                raise ValueError("conversion output must be separate from input")
    if "evaluate" in args.stages:
        for name in ("launcher", "binary", "bench_script"):
            require_file(plan[name])
        bench_code(plan["bench_script"], plan["run_dir"])
        require_model(plan["baseline"], converted=True)
        if "convert" not in args.stages:
            require_model(plan["candidate"], converted=True)
        if plan["candidate"] == plan["baseline"]:
            raise ValueError("baseline and candidate must be different directories")
        require_free_port(args.port)
        if os.name == "nt":
            taskkill = shutil.which("taskkill")
            if not taskkill:
                raise ValueError("taskkill required to stop the owned server process tree")
            plan["taskkill"] = Path(taskkill).resolve()
    if "graph" in args.stages:
        require_file(plan["graph_script"])
    return plan


def run(command, plan):
    print("Launching: " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(plan["engine_root"]), check=True)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def compare_results(baseline, candidate):
    suites = {}
    for suite in ("hard", "base"):
        before, after = baseline[suite], candidate[suite]
        b, c = before["summary"], after["summary"]
        old = {r["id"]: r["passed"] for r in before.get("rows", []) if type(r.get("passed")) is bool}
        new = {r["id"]: r["passed"] for r in after.get("rows", []) if type(r.get("passed")) is bool}
        common = old.keys() & new.keys()
        suites[suite] = {"original": b, "candidate": c, "scope": "retention_only",
                         "score_delta": c["passed"] - b["passed"],
                         "per_task_regressions": sorted(k for k in common if old[k] and not new[k]),
                         "per_task_improvements": sorted(k for k in common if new[k] and not old[k]),
                         "task_comparison_complete": bool(old) and old.keys() == new.keys()
                            and len(old) == b["total"] == c["total"],
                         "missing_candidate_tasks": sorted(old.keys() - new.keys()),
                         "missing_baseline_tasks": sorted(new.keys() - old.keys())}
    delta = sum(s["score_delta"] for s in suites.values())
    regression = any(s["score_delta"] < 0 or s["per_task_regressions"] for s in suites.values())
    return {"scope": "retention_only", "promotion_eligible": False, "proven_improvement": False,
            "recommendation": "keep original; no change", "regression": regression,
            "score_delta": delta, "suites": suites,
            "note": ("Tie: no proven improvement. " if delta == 0 else "") +
                    "Legacy hard/base benchmarks overlap training; retention only, never promotion evidence. "
                    "Independent held-out evaluation is required."}


def stop_server(process, plan):
    if os.name == "nt":
        stopped = subprocess.run([str(plan["taskkill"]), "/PID", str(process.pid), "/T", "/F"],
                                 capture_output=True)
        if stopped.returncode and process.poll() is None:
            raise RuntimeError("could not stop owned server tree; refusing further launches")
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    process.wait(timeout=30)
    # Do not start the next model while the previous listener is still alive.
    require_free_port(plan["port"])


def evaluate_model(plan, model, label):
    port = plan["port"]
    require_free_port(port)
    model_id = plan["run_id"] + "-" + label
    env = dict(os.environ, COLI_VULKAN="1", COLI_ENGINE=str(plan["binary"]),
               PYTHONDONTWRITEBYTECODE="1")
    command = [str(plan["engine_python"]), str(plan["launcher"]), "serve", "--model", str(model),
               "--auto-tier", "--policy", "quality", "--ctx", "8192", "--ngen", "700",
               "--host", "127.0.0.1", "--port", str(port), "--api-key", "local", "--model-id", model_id]
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (plan["run_dir"] / f"serve_{label}.log").open("xb") as log:
        require_free_port(port)
        process = subprocess.Popen(command, cwd=str(plan["engine_root"]), env=env,
                                   stdout=log, stderr=log, **options)
        try:
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited; see serve_{label}.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        ready = response.status == 200
                    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                                                     headers={"Authorization": "Bearer local"})
                    with urllib.request.urlopen(request, timeout=2) as response:
                        owned = model_id in {m["id"] for m in json.load(response)["data"]}
                    if ready and owned and process.poll() is None:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("owned server did not become healthy in 300 seconds")
                time.sleep(1)
            for suite in ("hard", "base"):
                # Legacy bench has no results flag. A worker redirects RESULTS before
                # execution, avoiding any files/edits in the engine checkout.
                run([plan["engine_python"], "-B", Path(__file__).resolve(), "--bench-worker",
                     plan["bench_script"], plan["run_dir"],
                     "--base", f"http://127.0.0.1:{port}/v1", "--model", model_id,
                     "--label", f"{label}_{suite}", "--suite", suite,
                     "--api-key", "local", "--max-tokens", "700", "--no-think"], plan)
        finally:
            stop_server(process, plan)


def main(argv=None):
    args = parse_args(argv)
    plan = preflight(args)
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if args.preflight_only:
        return
    plan["results_dir"].mkdir(parents=True, exist_ok=True)
    plan["run_dir"].mkdir()  # Atomic unique reservation; never reuse a run.
    write_json(plan["run_dir"] / "run_plan.json", plan)
    if "merge" in args.stages:
        require_new(plan["merged"])
        run([plan["python"], plan["merge_script"], "--base", plan["base"],
             "--adapter", plan["adapter"], "--out", plan["merged"], "--scaling", plan["scaling"]], plan)
    if "convert" in args.stages:
        # The legacy converter permits existing directories: claim a fresh one first.
        plan["candidate"].mkdir(parents=True, exist_ok=False)
        run([plan["engine_python"], plan["convert_script"], "--model", plan["merged"],
             "--out", plan["candidate"], "--ebits", "4", "--gs", "64", "--no-readme"], plan)
    if "evaluate" in args.stages:
        evaluate_model(plan, plan["baseline"], "qwen36")
        evaluate_model(plan, plan["candidate"], "finetune")
        def load(label):
            return {s: json.loads((plan["run_dir"] / f"bench_best_{label}_{s}.json").read_text(encoding="utf-8"))
                    for s in ("hard", "base")}
        verdict = compare_results(load("qwen36"), load("finetune"))
        write_json(plan["run_dir"] / "finetune_verdict.json", verdict)
        print(json.dumps(verdict, indent=2), flush=True)
    if "graph" in args.stages:
        run([plan["python"], plan["graph_script"], "--results-dir", plan["run_dir"]], plan)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bench-worker":
        bench, results = (Path(p).resolve() for p in sys.argv[2:4])
        sys.argv = [str(bench), *sys.argv[4:]]
        exec(bench_code(bench, results), {"__name__": "__main__", "__file__": str(bench)})
    else:
        main()
