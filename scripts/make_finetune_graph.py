"""Before/after graph for the Qwen3.6-35B AirLLM LoRA fine-tune.

Reads baseline results/bench_best_qwen36_{hard,base}.json and candidate
results/bench_best_finetune_{hard,base}.json plus the training log, writes
results/bench_finetune_comparison.png and prints the winner.

Run: .venv/Scripts/python.exe scripts/make_finetune_graph.py
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

BG = "#15120F"
PANEL = "#1B1714"
BORDER = "#3C322A"
TEXT = "#F2EEE8"
MUTED = "#A99D91"
ACCENT = "#E9865D"
GREEN = "#5EC98F"
RED = "#EF615C"


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def loss_curve():
    steps, losses, avg = [], [], []
    path = os.path.join(RESULTS, "airllm_train.log")
    if not os.path.exists(path):
        return steps, losses, avg
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"step (\d+) epoch \d+ loss=([\d.]+) avg5=([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            avg.append(float(m.group(3)))
    return steps, losses, avg


def main():
    bh, bb = load("bench_best_qwen36_hard.json"), load("bench_best_qwen36_base.json")
    ch, cb = load("bench_best_finetune_hard.json"), load("bench_best_finetune_base.json")
    bs, cs = bh["summary"], ch["summary"]
    bs2, cs2 = bb["summary"], cb["summary"]

    base_total = bs["passed"] + bs2["passed"]
    cand_total = cs["passed"] + cs2["passed"]
    n_total = bs["total"] + bs2["total"]
    if cand_total > base_total:
        verdict, color = "FINE-TUNED WINS", GREEN
    elif cand_total < base_total:
        verdict, color = "ORIGINAL WINS (regression - keep original)", RED
    else:
        verdict, color = "TIE ON SCORES - fine-tuned kept (no regression)", ACCENT

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
        "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#2A231D", "font.size": 10.5, "axes.titlesize": 12,
    })
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22,
                          left=0.06, right=0.97, top=0.88, bottom=0.09)

    fig.suptitle("Qwen3.6-35B-A3B: AirLLM streamed LoRA fine-tune (rank 16, 20 verified SFT pairs, 2 epochs)",
                 fontsize=13.5, color=TEXT, y=0.97)
    fig.text(0.5, 0.925, verdict, ha="center", fontsize=12.5, color=color, weight="bold")

    ax = fig.add_subplot(gs[0, 0])
    labels = ["hard suite (12)", "base suite (10)"]
    x = range(2)
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], [bs["passed"], bs2["passed"]], w,
                color=MUTED, label="original int4 gs64")
    b2 = ax.bar([i + w / 2 for i in x], [cs["passed"], cs2["passed"]], w,
                color=ACCENT, label="fine-tuned candidate")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.15,
                    f"{int(r.get_height())}", ha="center", fontsize=10, color=TEXT)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 13.5); ax.set_ylabel("execution-verified passes")
    ax.set_title("Pass@1 (greedy, no-think, hidden unit tests)")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    steps, losses, avg = loss_curve()
    if steps:
        ax.plot(steps, losses, "o-", color=ACCENT, ms=4, label="step loss")
        ax.plot(steps, avg, "-", color=GREEN, lw=1.6, label="avg5")
        ax.set_xlabel("training step"); ax.set_ylabel("loss")
        ax.set_title(f"Streamed LoRA training loss ({losses[0]:.2f} -> {avg[-1]:.2f}, ~2 GB VRAM)")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    note = (
        f"Original: hard {bs['passed']}/{bs['total']} @ {bs['tokens_per_s']} tok/s (healthy Vulkan), "
        f"base {bs2['passed']}/{bs2['total']} @ {bs2['tokens_per_s']} tok/s.\n"
        f"Candidate: hard {cs['passed']}/{cs['total']} @ {cs['tokens_per_s']} tok/s, "
        f"base {cs2['passed']}/{cs2['total']} @ {cs2['tokens_per_s']} tok/s "
        f"(serve backend at bench time; speed is backend-dependent, scores are not).\n"
        "Caveat: the hard suite overlaps the 20 verified SFT training tasks (retention check); "
        "the base suite is mostly held out (generalization check).\n"
        "Promotion rule: candidate replaces qwen36-35b only with zero regressions; "
        "original container is kept as backup either way."
    )
    ax.text(0.01, 0.95, note, va="top", fontsize=10, color=MUTED, linespacing=1.7)

    out = os.path.join(RESULTS, "bench_finetune_comparison.png")
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"saved {out}")
    print(f"{verdict}: original {base_total}/{n_total}, candidate {cand_total}/{n_total}")


if __name__ == "__main__":
    main()