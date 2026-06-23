#!/usr/bin/env python3
"""Render a publication-style zero-shot detection figure from modal answer JSONL."""

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

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 220, 0),
    "bus": (0, 180, 255),
    "truck": (255, 140, 0),
    "person": (255, 120, 0),  # blue in BGR
    "bicycle": (200, 0, 200),
    "motorcycle": (0, 200, 200),
}

DISPLAY_ORDER = ["car", "bus", "truck", "person", "bicycle", "motorcycle"]


def draw_box(
    img: np.ndarray,
    box: list[float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    text = label
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    text_th = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, text_th)
    tag_y1 = max(0, y1 - th - baseline - 6)
    tag_y2 = tag_y1 + th + baseline + 6
    tag_x2 = min(img.shape[1], x1 + tw + 10)
    cv2.rectangle(img, (x1, tag_y1), (tag_x2, tag_y2), color, -1, cv2.LINE_AA)
    cv2.putText(
        img,
        text,
        (x1 + 5, tag_y2 - baseline - 3),
        font,
        scale,
        (255, 255, 255),
        text_th,
        cv2.LINE_AA,
    )


def draw_panel(img: np.ndarray, preds: dict[str, list], counts: dict[str, int]) -> None:
    overlay = img.copy()
    panel_h, panel_w = 150, 420
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 38
    cv2.putText(img, "LocateAnything-3B (Zero-Shot)", (24, y), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    y += 28
    cv2.putText(img, "D2-City Smoke Test Frame", (24, y), font, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    y += 26

    label_names = {
        "car": "Car",
        "bus": "Bus",
        "truck": "Truck",
        "person": "Person",
        "bicycle": "Bicycle",
        "motorcycle": "Motorcycle",
    }
    summary = " + ".join(
        f"{counts[c]} {label_names[c]}{'s' if counts[c] != 1 and c != 'person' else ''}"
        for c in DISPLAY_ORDER
        if counts.get(c, 0)
    )
    cv2.putText(img, summary + " Detected", (24, y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    y += 30

    lx = 24
    for cls in DISPLAY_ORDER:
        if not counts.get(cls):
            continue
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(img, (lx, y - 10), (lx + 14, y + 4), color, 2, cv2.LINE_AA)
        cv2.putText(img, cls, (lx + 22, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        lx += 90


def summary_counts(preds: dict[str, list]) -> dict[str, int]:
    return {k: len(v) for k, v in preds.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render annotated detection figure")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "config" / "d2city_eval.yaml"),
    )
    parser.add_argument("--index", type=int, default=0, help="Line index in modal answer JSONL")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR.parent / "results" / "D2City_val" / "modal" / "zero_shot_smoke_test_frame.png"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    answer_jsonl = get_modal_output_jsonl(cfg)
    image_root = get_d2city_processed_dir(cfg)

    with open(answer_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.index:
                entry = json.loads(line)
                break
        else:
            raise SystemExit(f"Index {args.index} not found in {answer_jsonl}")

    image_path = image_root / entry["image_path"]
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read: {image_path}")

    preds = entry.get("extracted_predictions", {})
    counts = summary_counts(preds)

    for cls in DISPLAY_ORDER:
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        for box in preds.get(cls, []):
            draw_box(img, box, cls, color)

    draw_panel(img, preds, counts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Wrote {out}")
    print(f"Source: {image_path}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
