#!/usr/bin/env python3
"""Line chart: Precision / Recall / F1 vs IoU threshold (D2-City zero-shot results)."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

EVAL_JSON = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "D2City_val"
    / "modal"
    / "eval_results.json"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def load_curve(eval_path: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    data = json.loads(eval_path.read_text())
    ious = sorted((float(k) for k in data.keys()))
    precisions, recalls, f1s = [], [], []
    for iou in ious:
        key = f"{iou:g}" if iou in (0.5, 0.55) else str(iou)
        # keys stored as '0.5', '0.55', ...
        block = data.get(str(iou)) or data.get(f"{iou:.2f}")
        if block is None:
            for k in data:
                if abs(float(k) - iou) < 1e-6:
                    block = data[k]
                    break
        metrics = block["basic_metrics"]["common_object_detection_D2City"]
        p = mean(metrics["precisions"])
        r = mean(metrics["recalls"])
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return ious, precisions, recalls, f1s


def main() -> None:
    ious, precisions, recalls, f1s = load_curve(EVAL_JSON)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.plot(ious, f1s, "o-", color="#76B900", linewidth=2.5, markersize=7, label="F1", zorder=3)
    ax.plot(ious, recalls, "s--", color="#0066CC", linewidth=1.8, markersize=5, label="Recall", alpha=0.9)
    ax.plot(ious, precisions, "^--", color="#E87722", linewidth=1.8, markersize=5, label="Precision", alpha=0.9)

    # Highlight IoU 0.5 and 0.9
    for iou_target, color, label in [(0.5, "#76B900", "IoU 0.50"), (0.9, "#CC0000", "IoU 0.90")]:
        idx = min(range(len(ious)), key=lambda i: abs(ious[i] - iou_target))
        ax.axvline(ious[idx], color=color, linestyle=":", alpha=0.45, linewidth=1.2)
        ax.annotate(
            f"{label}\nF1={f1s[idx]:.3f}",
            xy=(ious[idx], f1s[idx]),
            xytext=(ious[idx] + 0.03, f1s[idx] + 0.08),
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1),
        )

    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        "Zero-Shot LocateAnything-3B on D2-City (499 frames)\n"
        "Coarse localization holds; tight boxes collapse",
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(ious)
    ax.set_xticklabels([f"{v:.2f}" for v in ious], rotation=45, ha="right")
    ax.set_ylim(0, 0.88)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    caption = (
        "Macro-averaged per-frame Precision, Recall, and F1 from official other_metric.py. "
        "One of 500 frames excluded (HTTP 408 timeout). "
        f"IoU 0.50: F1={f1s[0]:.3f} | IoU 0.90: F1={f1s[ious.index(0.9)]:.3f} | "
        f"mIoU F1={np.mean(f1s):.3f}"
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5, color="#555555")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "d2city_iou_threshold_curve.png"
    out_svg = OUT_DIR / "d2city_iou_threshold_curve.svg"
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
