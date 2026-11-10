"""Plot explicit hard/base retention results, never overwrite a historical graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

BG = "#15120F"
PANEL = "#1B1714"
BORDER = "#3C322A"
TEXT = "#F2EEE8"
MUTED = "#A99D91"
ACCENT = "#E9865D"
GREEN = "#5EC98F"
RED = "#EF615C"


def load(results, name):
    return json.loads((results / name).read_text(encoding="utf-8"))


def loss_curve(results):
    steps, losses, avg = [], [], []
    path = results / "airllm_train.log"
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"step (\d+) epoch \d+ loss=([\d.]+) avg\d+=([\d.]+)", line)
            if match:
                steps.append(int(match[1]))
                losses.append(float(match[2]))
                avg.append(float(match[3]))
    return steps, losses, avg


def retention_title(verdict):
    if verdict["regression"]:
        return "RETENTION REGRESSION - keep original", RED
    if verdict["score_delta"] == 0:
        return "TIE - no proven improvement; no change", ACCENT
    return "RETENTION SCORE INCREASE ONLY - no promotion; keep original", GREEN


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, help="directory containing the four benchmark JSON files")
    args = ap.parse_args(argv)
    results = Path(args.results_dir).resolve()
    out = results / "bench_finetune_comparison.png"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing graph: {out}")
    baseline = {s: load(results, f"bench_best_qwen36_{s}.json") for s in ("hard", "base")}
    candidate = {s: load(results, f"bench_best_finetune_{s}.json") for s in ("hard", "base")}
    from run_finetune_eval import compare_results
    verdict = compare_results(baseline, candidate)
    title, color = retention_title(verdict)
    bs, bs2 = (baseline[s]["summary"] for s in ("hard", "base"))
    cs, cs2 = (candidate[s]["summary"] for s in ("hard", "base"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
        "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2A231D", "font.size": 10.5, "axes.titlesize": 12,
    })
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22,
                          left=0.06, right=0.97, top=0.88, bottom=0.09)
    fig.suptitle("Qwen3.6-35B-A3B: AirLLM LoRA retention comparison",
                 fontsize=13.5, color=TEXT, y=0.97)
    fig.text(0.5, 0.925, title, ha="center", fontsize=12.5, color=color, weight="bold")
    ax = fig.add_subplot(gs[0, 0])
    x, width = range(2), 0.36
    bars = [ax.bar([i - width / 2 for i in x], [bs["passed"], bs2["passed"]], width,
                   color=MUTED, label="original"),
            ax.bar([i + width / 2 for i in x], [cs["passed"], cs2["passed"]], width,
                   color=ACCENT, label="fine-tuned candidate")]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    str(int(bar.get_height())), ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"hard ({bs['total']}/{cs['total']} tasks)",
                        f"base ({bs2['total']}/{cs2['total']} tasks)"])
    ax.set_ylim(0, max(s["total"] for s in (bs, bs2, cs, cs2)) + 1.5)
    ax.set_ylabel("execution-verified passes")
    ax.set_title("Legacy overlapping suites: retention only")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    steps, losses, avg = loss_curve(results)
    if steps:
        ax.plot(steps, losses, "o-", color=ACCENT, ms=4, label="step loss")
        ax.plot(steps, avg, "-", color=GREEN, lw=1.6, label="logged running mean")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.set_title("Streamed LoRA training loss")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No training loss log in this run", ha="center")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    regressions = [f"{suite}:{task}" for suite, data in verdict["suites"].items()
                   for task in data["per_task_regressions"]]
    complete = all(s["task_comparison_complete"] for s in verdict["suites"].values())
    note = (
        f"Original: hard {bs['passed']}/{bs['total']}; base {bs2['passed']}/{bs2['total']}. "
        f"Candidate: hard {cs['passed']}/{cs['total']}; base {cs2['passed']}/{cs2['total']}.\n"
        "Both legacy suites overlap training: retention only, not held-out generalization evidence.\n"
        "No automatic promotion. Ties show no proven improvement; keep the original unchanged.\n"
        f"Per-task regressions: {', '.join(regressions) or 'none observed'}; "
        f"task coverage {'complete' if complete else 'incomplete/unavailable'}.\n"
        "Independent held-out evaluation is required; historic runs may differ in configuration/backend."
    )
    ax.text(0.01, 0.95, note, va="top", fontsize=10, color=MUTED, linespacing=1.7, wrap=True)
    try:
        with out.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150, facecolor=BG)
    finally:
        plt.close(fig)
    print(f"saved {out}\n{title}")


if __name__ == "__main__":
    main()
