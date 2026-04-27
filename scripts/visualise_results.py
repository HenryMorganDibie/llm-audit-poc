"""
scripts/visualise_results.py

Generates charts from the redacted audit results.
Run: python scripts/visualise_results.py
Output: results/figures/
"""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BLUE   = "#1F4E79"
LBLUE  = "#4A90C4"
RED    = "#C00000"
ORANGE = "#E06C00"
GREEN  = "#375623"
GRAY   = "#888888"
LGRAY  = "#F2F2F2"

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

with open("results/eval_results_redacted.json") as f:
    data = json.load(f)

models = data["models"]
ckpts  = data["checkpoints"]


# ── 1. BFCL Accuracy Comparison ──────────────────────────────────────────────
def plot_accuracy_comparison():
    labels = ["7B Baseline", "3B Baseline", "Fine-tuned\n7B LoRA", "Filtered\nRetrain"]
    strict = [
        models["baseline_7b"]["bfcl_strict_pct"],
        models["baseline_3b"]["bfcl_strict_pct"],
        models["finetuned_7b_lora_r32"]["bfcl_strict_pct"],
        models["finetuned_7b_lora_filtered_1epoch"]["bfcl_strict_pct"],
    ]
    loose = [
        models["baseline_7b"]["bfcl_loose_pct"],
        models["baseline_3b"]["bfcl_loose_pct"],
        models["finetuned_7b_lora_r32"]["bfcl_loose_pct"],
        models["finetuned_7b_lora_filtered_1epoch"]["bfcl_loose_pct"],
    ]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LGRAY)

    b1 = ax.bar(x - w/2, strict, w, label="Strict", color=[BLUE, BLUE, RED, ORANGE], alpha=0.9, zorder=3)
    b2 = ax.bar(x + w/2, loose,  w, label="Loose",  color=[LBLUE, LBLUE, "#E07070", "#F0A060"], alpha=0.7, zorder=3)

    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("BFCL Accuracy (%)", fontsize=11)
    ax.set_title("BFCL Python Subset Accuracy — All Models", fontsize=13, fontweight="bold", color=BLUE, pad=14)
    ax.axhline(90, color=GREEN, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2, label="Production threshold (90%)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", color="white", linewidth=1.5, zorder=1)
    ax.spines[["top","right","left"]].set_visible(False)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#333333")

    plt.tight_layout()
    plt.savefig(OUT / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: accuracy_comparison.png")


# ── 2. Regression Decomposition ──────────────────────────────────────────────
def plot_regression_decomposition():
    decomp = data["regression_decomposition"]
    categories = ["Genuine\nRegression", "Prefix\nArtifact"]
    values     = [decomp["genuine_regression_pp"], decomp["prefix_artifact_pp"]]
    colors     = [RED, ORANGE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("white")

    # Waterfall
    ax1.set_facecolor(LGRAY)
    baseline = models["baseline_7b"]["bfcl_strict_pct"]
    finetuned= models["finetuned_7b_lora_r32"]["bfcl_strict_pct"]

    ax1.bar(["Baseline\n7B"], [baseline], color=BLUE, alpha=0.9, zorder=3, width=0.5)
    ax1.bar(["Fine-tuned\n7B"], [finetuned], color=RED, alpha=0.9, zorder=3, width=0.5)
    ax1.annotate("", xy=(1, finetuned + 2), xytext=(0, baseline - 2),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax1.text(0.5, (baseline + finetuned)/2 + 3, f"−{decomp['total_regression_pp']}pp",
             ha="center", fontsize=13, color=RED, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("BFCL Strict Accuracy (%)", fontsize=10)
    ax1.set_title("Accuracy Drop After Fine-Tuning", fontsize=11, fontweight="bold", color=BLUE)
    ax1.grid(axis="y", color="white", linewidth=1.5, zorder=1)
    ax1.spines[["top","right","left"]].set_visible(False)

    # Pie decomposition
    ax2.pie(values, labels=categories, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax2.set_title("Regression Decomposition\n(genuine vs scorer artifact)", fontsize=11, fontweight="bold", color=BLUE)

    plt.suptitle("50pp BFCL Regression: What's Real vs What's the Scorer", fontsize=12,
                 fontweight="bold", color="#333333", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "regression_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: regression_decomposition.png")


# ── 3. Checkpoint Trajectory ─────────────────────────────────────────────────
def plot_checkpoint_trajectory():
    steps  = [c["step"] for c in ckpts]
    scores = [c["bfcl_strict_pct"] for c in ckpts]
    epochs = [c["epoch_approx"] for c in ckpts]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LGRAY)

    ax.plot(steps, scores, "o-", color=RED, linewidth=2.5, markersize=8, zorder=4)
    ax.axhline(models["baseline_3b"]["bfcl_strict_pct"], color=GREEN, linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"3B Baseline ({models['baseline_3b']['bfcl_strict_pct']}%)")
    ax.axhline(models["baseline_7b"]["bfcl_strict_pct"], color=BLUE, linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"7B Baseline ({models['baseline_7b']['bfcl_strict_pct']}%)")

    for s, sc, e in zip(steps, scores, epochs):
        ax.annotate(f"{sc}%\n(~{e} ep)", (s, sc), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="#333333")

    ax.fill_between(steps, scores, alpha=0.15, color=RED)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("BFCL Strict Accuracy (%)", fontsize=11)
    ax.set_title("Checkpoint Trajectory — Regression Baked Before Epoch 1", fontsize=12,
                 fontweight="bold", color=BLUE, pad=12)
    ax.set_ylim(30, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", color="white", linewidth=1.5, zorder=1)
    ax.spines[["top","right","left"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "checkpoint_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: checkpoint_trajectory.png")


# ── 4. Throughput & Cost Comparison ──────────────────────────────────────────
def plot_throughput_cost():
    names  = ["3B FP16\n(Recommended)", "3B FP8", "7B LoRA\nFP8"]
    tputs  = [6075, 6780, 4154]
    costs  = [0.14, 0.14, 0.23]
    colors = [GREEN, LBLUE, RED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("white")

    for ax, vals, ylabel, title, fmt in [
        (ax1, tputs, "Tokens / second", "Throughput at Concurrency 32", "{:,.0f}"),
        (ax2, costs, "Cost ($/M tokens)", "Estimated Cost per Million Tokens", "${:.2f}"),
    ]:
        ax.set_facecolor(LGRAY)
        bars = ax.bar(names, vals, color=colors, alpha=0.88, zorder=3, width=0.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", color=BLUE)
        ax.grid(axis="y", color="white", linewidth=1.5, zorder=1)
        ax.spines[["top","right","left"]].set_visible(False)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                    fmt.format(h), ha="center", va="bottom", fontsize=10, color="#333333")

    plt.suptitle("Serving Performance — 1× H100 (80GB)", fontsize=12,
                 fontweight="bold", color="#333333")
    plt.tight_layout()
    plt.savefig(OUT / "throughput_cost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: throughput_cost.png")


if __name__ == "__main__":
    plot_accuracy_comparison()
    plot_regression_decomposition()
    plot_checkpoint_trajectory()
    plot_throughput_cost()
    print(f"\nAll figures saved to {OUT}/")
