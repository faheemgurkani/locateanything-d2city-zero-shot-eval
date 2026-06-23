#!/usr/bin/env python3
"""GT vs prediction overlay strip for qualitative localization analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_d2city_processed_dir, get_modal_output_jsonl, load_config  # noqa: E402

GT_COLOR = (50, 220, 50)      # green BGR
PRED_COLOR = (80, 80, 255)    # red BGR
GT_LABEL = "Ground truth"
PRED_LABEL = "Prediction"


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def frame_match_stats(entry: dict) -> dict:
    """Per-frame best-match IoU stats across all classes."""
    gt = entry.get("gt", {})
    pred = entry.get("extracted_predictions", {})
    ious: list[float] = []
    loose_hits = 0
    tight_hits = 0
    total_gt = 0

    for cls, gt_boxes in gt.items():
        pred_boxes = pred.get(cls, [])
        for g in gt_boxes:
            total_gt += 1
            best = max((box_iou(g, p) for p in pred_boxes), default=0.0)
            ious.append(best)
            if best >= 0.5:
                loose_hits += 1
            if best >= 0.9:
                tight_hits += 1

    mean_iou = float(np.mean(ious)) if ious else 0.0
    min_iou = float(np.min(ious)) if ious else 0.0
    return {
        "mean_best_iou": mean_iou,
        "min_best_iou": min_iou,
        "loose_recall": loose_hits / total_gt if total_gt else 0,
        "tight_recall": tight_hits / total_gt if total_gt else 0,
        "total_gt": total_gt,
        "loose_hits": loose_hits,
        "tight_hits": tight_hits,
    }


def pick_example_frames(entries: list[dict], n: int = 3) -> list[tuple[dict, str]]:
    """Select diverse frames: good loose match, loose-but-not-tight, weak overall."""
    scored = []
    for e in entries:
        if not e.get("gt"):
            continue
        stats = frame_match_stats(e)
        if stats["total_gt"] < 2:
            continue
        scored.append((e, stats))

    if not scored:
        return []

    # Good loose match: high loose recall, decent mean IoU
    good_loose = max(
        scored,
        key=lambda x: (x[1]["loose_recall"], x[1]["mean_best_iou"]),
    )

    # Loose OK but tight fails: good loose recall, poor tight recall
    loose_not_tight = max(
        scored,
        key=lambda x: x[1]["loose_recall"] - x[1]["tight_recall"] * 2 + x[1]["total_gt"] * 0.01,
    )

    # Visible offset: moderate mean IoU, low min IoU (some boxes badly placed)
    offset = max(
        scored,
        key=lambda x: x[1]["mean_best_iou"] - x[1]["min_best_iou"],
    )

    picks: list[tuple[dict, str]] = []
    seen_paths: set[str] = set()

    def add(item: tuple[dict, dict], caption: str) -> None:
        path = item[0]["image_path"]
        if path in seen_paths:
            return
        seen_paths.add(path)
        s = item[1]
        cap = (
            f"{caption}\n"
            f"loose@{0.5}: {s['loose_hits']}/{s['total_gt']}  "
            f"tight@{0.9}: {s['tight_hits']}/{s['total_gt']}  "
            f"mean IoU: {s['mean_best_iou']:.2f}"
        )
        picks.append((item[0], cap))

    add(good_loose, "Strong loose match")
    add(loose_not_tight, "Right region, loose boxes")
    add(offset, "Visible box offset")

    if len(picks) < n:
        for e, s in sorted(scored, key=lambda x: -x[1]["total_gt"]):
            add((e, s), "Urban traffic scene")
            if len(picks) >= n:
                break
    return picks[:n]


def draw_boxes(img: np.ndarray, boxes_by_class: dict, color: tuple[int, int, int], prefix: str) -> None:
    for cls, boxes in boxes_by_class.items():
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(
                img,
                f"{prefix}:{cls}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )


def render_panel(
    img: np.ndarray,
    entry: dict,
    caption: str,
    panel_w: int = 520,
    panel_h: int = 320,
) -> np.ndarray:
    """Side-by-side GT | Prediction."""
    h, w = img.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    nw, nh = int(w * scale), int(h * scale)
    base = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def scale_boxes(boxes_by_class: dict) -> dict:
        out: dict = {}
        for cls, boxes in boxes_by_class.items():
            out[cls] = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in boxes]
        return out

    gt_scaled = scale_boxes(entry.get("gt", {}))
    pred_scaled = scale_boxes(entry.get("extracted_predictions", {}))

    gt_img = base.copy()
    pred_img = base.copy()
    draw_boxes(gt_img, gt_scaled, GT_COLOR, "GT")
    draw_boxes(pred_img, pred_scaled, PRED_COLOR, "Pred")

    gap = 8
    label_h = 22
    cap_h = 36
    canvas = np.full((nh + label_h + cap_h, nw * 2 + gap, 3), 245, dtype=np.uint8)
    canvas[label_h : label_h + nh, 0:nw] = gt_img
    canvas[label_h : label_h + nh, nw + gap : nw + gap + nw] = pred_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Ground truth", (8, 16), font, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "LocateAnything prediction", (nw + gap + 8, 16), font, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, caption, (8, nh + label_h + 22), font, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def build_strip(panels: list[np.ndarray], pad: int = 10, header_h: int = 56) -> np.ndarray:
    max_w = max(p.shape[1] for p in panels)
    total_h = header_h + sum(p.shape[0] for p in panels) + (len(panels) + 1) * pad
    canvas = np.full((total_h, max_w + 2 * pad, 3), 250, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "Qualitative localization: ground truth vs zero-shot predictions",
        (pad, 24),
        font,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Green = GT  |  Red = prediction  |  Loose IoU matches may still fail at tight thresholds",
        (pad, 46),
        font,
        0.45,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    y = header_h + pad
    for panel in panels:
        x = pad + (max_w - panel.shape[1]) // 2
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        y += panel.shape[0] + pad
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GT vs pred comparison strip")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "config" / "d2city_eval.yaml"),
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR.parent / "results" / "figures" / "d2city_gt_pred_comparison.png"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    answer_jsonl = get_modal_output_jsonl(cfg)
    image_root = get_d2city_processed_dir(cfg)

    entries = []
    with open(answer_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    picks = pick_example_frames(entries, n=args.n)
    if not picks:
        raise SystemExit("No suitable frames found in modal answer JSONL")

    panels = []
    for entry, caption in picks:
        img_path = image_root / entry["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        panels.append(render_panel(img, entry, caption))

    strip = build_strip(panels)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Wrote {out}")
    for entry, cap in picks:
        print(f"  {entry['image_path']}")


if __name__ == "__main__":
    main()
