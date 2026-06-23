#!/usr/bin/env python3
"""Latency distribution chart from Modal batch eval per-frame timings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_modal_output_jsonl, load_config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
ADAS_TARGET_FPS = 10  # reference line for production ADAS


def load_latencies(answer_jsonl: Path) -> np.ndarray:
    values = []
    with open(answer_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ms = entry.get("latency_ms")
            if ms is not None:
                values.append(float(ms))
    return np.array(values)


def main() -> None:
    cfg = load_config(SCRIPT_DIR.parent / "config" / "d2city_eval.yaml")
    answer_jsonl = get_modal_output_jsonl(cfg)
    lat_ms = load_latencies(answer_jsonl)
    if len(lat_ms) == 0:
        raise SystemExit(f"No latency_ms found in {answer_jsonl}")

    mean_ms = float(np.mean(lat_ms))
    median_ms = float(np.median(lat_ms))
    max_ms = float(np.max(lat_ms))
    min_ms = float(np.min(lat_ms))
    p90_ms = float(np.percentile(lat_ms, 90))
    fps_mean = 1000.0 / mean_ms
    fps_median = 1000.0 / median_ms

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 11,
        }
    )

    fig = plt.figure(figsize=(10, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.28)

    # --- Left: histogram ---
    ax0 = fig.add_subplot(gs[0, 0])
    bins = np.arange(300, 1650, 50)
    ax0.hist(
        lat_ms,
        bins=bins,
        color="#76B900",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )
    for val, color, label in [
        (median_ms, "#0066CC", f"Median {median_ms:.0f} ms"),
        (mean_ms, "#E87722", f"Mean {mean_ms:.0f} ms"),
        (max_ms, "#CC0000", f"Max {max_ms:.0f} ms"),
    ]:
        ax0.axvline(val, color=color, linestyle="--", linewidth=1.6, alpha=0.9, label=label)

    ax0.legend(loc="upper right", fontsize=8.5, frameon=True)

    ax0.set_xlabel("Per-frame latency (ms)")
    ax0.set_ylabel("Frame count")
    ax0.set_title("Latency distribution (499 scored frames)", fontweight="bold", pad=10)
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # --- Right: summary bars + ADAS reference ---
    ax1 = fig.add_subplot(gs[0, 1])
    labels = ["Mean", "Median", "P90", "Max"]
    values = [mean_ms, median_ms, p90_ms, max_ms]
    colors = ["#E87722", "#0066CC", "#888888", "#CC0000"]
    bars = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Summary statistics", fontweight="bold", pad=10)
    ax1.set_ylim(0, max(values) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Throughput annotation box
    text = (
        f"Effective throughput\n"
        f"Mean: {fps_mean:.1f} FPS\n"
        f"Median: {fps_median:.1f} FPS\n\n"
        f"ADAS target: >{ADAS_TARGET_FPS} FPS\n"
        f"(not met in this setup)"
    )
    ax1.text(
        0.5,
        0.97,
        text,
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E7", edgecolor="#E87722", alpha=0.95),
    )

    fig.suptitle(
        "Zero-Shot LocateAnything-3B — Modal L40S, Hybrid Mode\n"
        "~1.5 FPS: capability probe, not real-time ADAS",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    caption = (
        f"Per-frame server-side latency from batch eval ({len(lat_ms)} frames; 1 timeout excluded). "
        f"Wall time ~34 min for 500 frames. Modal L40S, hybrid generation mode. "
        f"Min latency {min_ms:.0f} ms."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8.5, color="#555555")

    fig.subplots_adjust(top=0.78, bottom=0.16)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "d2city_latency_distribution.png"
    out_svg = OUT_DIR / "d2city_latency_distribution.svg"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {out_png}")
    print(f"Mean {mean_ms:.1f} ms | Median {median_ms:.1f} ms | Max {max_ms:.1f} ms | FPS {fps_mean:.2f}")


if __name__ == "__main__":
    main()
