#!/usr/bin/env python3
"""Bar chart: LocateAnything-3B vs Rex-Omni-3B vs Qwen3-VL-4B (paper Tables 1 & 2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Paper Tables 1 & 2 — Vision-Language Models (3B-class comparison)
MODELS = ["LocateAnything-3B", "Rex-Omni-3B", "Qwen3-VL-4B"]
COLORS = ["#76B900", "#E87722", "#0066CC"]  # NVIDIA green, orange, blue

# Table 1: throughput (BPS) + mean F1 @ IoU (LVIS, COCO)
BPS = [12.7, 5.0, 1.1]
LVIS_MEAN = [50.7, 46.9, 43.5]
COCO_MEAN = [54.7, 52.9, 46.1]

# Table 2: mean F1 @ IoU (Dense200, VisDrone)
DENSE200_MEAN = [58.7, 58.3, 12.5]
VISDRONE_MEAN = [39.9, 35.8, 26.0]

MACRO_MEAN_F1 = [
    np.mean([LVIS_MEAN[i], COCO_MEAN[i], DENSE200_MEAN[i], VISDRONE_MEAN[i]])
    for i in range(3)
]

BENCHMARKS = ["LVIS", "COCO", "Dense200", "VisDrone"]
F1_BY_BENCH = np.array([LVIS_MEAN, COCO_MEAN, DENSE200_MEAN, VISDRONE_MEAN]).T


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "vlm_benchmark_comparison.png"
    out_svg = out_dir / "vlm_benchmark_comparison.svg"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 150,
        }
    )

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.28)

    x = np.arange(len(MODELS))
    bar_w = 0.55

    # --- Left: Throughput (BPS) ---
    ax0 = fig.add_subplot(gs[0, 0])
    bars0 = ax0.bar(x, BPS, width=bar_w, color=COLORS, edgecolor="white", linewidth=1.2)
    ax0.set_xticks(x)
    ax0.set_xticklabels(MODELS, rotation=12, ha="right")
    ax0.set_ylabel("Throughput (BPS)")
    ax0.set_title("Decoding Throughput", fontweight="bold", pad=12)
    ax0.set_ylim(0, max(BPS) * 1.22)
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    for bar, val in zip(bars0, BPS):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # --- Right: Mean F1 (macro + per-benchmark grouped) ---
    ax1 = fig.add_subplot(gs[0, 1])
    group_x = np.arange(len(BENCHMARKS) + 1)  # +1 for macro avg column
    group_w = 0.22
    offsets = [-group_w, 0, group_w]

    # Macro average column first visually — put it at the end instead for readability
    bench_labels = BENCHMARKS + ["Macro avg."]
    f1_extended = np.column_stack([F1_BY_BENCH, MACRO_MEAN_F1])

    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        ax1.bar(
            group_x + offsets[i],
            f1_extended[i],
            width=group_w,
            label=model,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )

    ax1.set_xticks(group_x)
    ax1.set_xticklabels(bench_labels)
    ax1.set_ylabel("Mean F1 @ IoU")
    ax1.set_title("Detection Quality (F1, mean over IoU thresholds)", fontweight="bold", pad=12)
    ax1.set_ylim(0, 68)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", frameon=True, fontsize=9, edgecolor="#cccccc")

    fig.suptitle(
        "LocateAnything-3B vs Rex-Omni-3B vs Qwen3-VL-4B\n"
        "Paper benchmark context (Tables 1 & 2)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    caption = (
        "Source: Locate Anything paper — LVIS & COCO (Table 1), Dense200 & VisDrone (Table 2). "
        "BPS = boxes per second. Macro avg. = mean of dataset-level mean F1 scores. "
        "Qwen3-VL-4B shown as closest-size general VLM baseline (~4B)."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=8.5, color="#555555", wrap=True)

    fig.subplots_adjust(top=0.78, bottom=0.14, wspace=0.28)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_svg}")
    print("\nSummary:")
    for m, bps, mf1 in zip(MODELS, BPS, MACRO_MEAN_F1):
        print(f"  {m}: BPS={bps}, macro mean F1={mf1:.1f}")


if __name__ == "__main__":
    main()
