#!/usr/bin/env python3
"""Montage of raw unannotated D²-City dashcam frames for article figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_d2city_processed_dir, load_config  # noqa: E402


def pick_from_videos(vid_dir: Path, n: int = 6) -> list[tuple[np.ndarray, str]]:
    """Extract frames from spread-out MP4 clips; keep brightness-diverse subset."""
    mp4s = sorted(vid_dir.glob("*.mp4"))
    if not mp4s:
        return []

    pool: list[tuple[float, np.ndarray, str]] = []
    step = max(1, len(mp4s) // (n * 3))
    for mp4 in mp4s[::step]:
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for frac in (0.15, 0.45, 0.75):
            idx = min(int(total * frac), max(total - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pool.append((float(np.mean(gray)), frame, mp4.stem[:12]))
        cap.release()

    if not pool:
        return []

    pool.sort(key=lambda x: x[0])
    indices = np.linspace(0, len(pool) - 1, n, dtype=int)
    chosen: list[tuple[np.ndarray, str]] = []
    seen: set[str] = set()
    for i in indices:
        _, img, name = pool[i]
        if name in seen:
            continue
        seen.add(name)
        chosen.append((img, name))

    if len(chosen) < n:
        for _, img, name in pool:
            if name not in seen:
                chosen.append((img, name))
                seen.add(name)
            if len(chosen) == n:
                break
    return chosen[:n]


def pick_diverse_frames(frames_dir: Path, n: int = 6) -> list[tuple[np.ndarray, str]]:
    """Pick frames spread across clips and brightness levels."""
    candidates: list[tuple[float, np.ndarray, str]] = []
    for vid_dir in sorted(frames_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        jpgs = sorted(vid_dir.glob("*.jpg"))
        if not jpgs:
            continue
        for idx in (0, len(jpgs) // 2, len(jpgs) - 1):
            p = jpgs[min(idx, len(jpgs) - 1)]
            img = cv2.imread(str(p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            candidates.append((float(np.mean(gray)), img, vid_dir.name[:12]))

    if len(candidates) < n:
        return [(img, name) for _, img, name in candidates]

    candidates.sort(key=lambda x: x[0])
    indices = np.linspace(0, len(candidates) - 1, n, dtype=int)
    chosen: list[tuple[np.ndarray, str]] = []
    seen_clips: set[str] = set()
    for i in indices:
        _, img, name = candidates[i]
        if name in seen_clips:
            continue
        seen_clips.add(name)
        chosen.append((img, name))

    if len(chosen) < n:
        for _, img, name in candidates:
            if name not in seen_clips:
                chosen.append((img, name))
                seen_clips.add(name)
            if len(chosen) == n:
                break
    return chosen[:n]


def resize_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (nw - w) // 2
    y0 = (nh - h) // 2
    return resized[y0 : y0 + h, x0 : x0 + w]


def scene_label(brightness: float) -> str:
    if brightness < 55:
        return "Low light / night"
    if brightness < 90:
        return "Overcast / haze"
    if brightness < 130:
        return "Urban traffic"
    return "Daytime / clear"


def build_montage(
    items: list[tuple[np.ndarray, str]],
    cols: int = 3,
    cell_w: int = 640,
    cell_h: int = 360,
    pad: int = 12,
    header_h: int = 72,
) -> np.ndarray:
    rows = (len(items) + cols - 1) // cols
    label_h = 28
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = header_h + rows * (cell_h + label_h) + (rows + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "D2-City: Dashcam Validation Split (raw frames, unannotated)",
        (pad, 32),
        font,
        0.85,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "1920x1080 source  |  diverse urban traffic  |  CVAT XML per-frame labels",
        (pad, 58),
        font,
        0.48,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    for i, (img, _clip) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = header_h + pad + r * (cell_h + label_h + pad)

        cell = resize_crop(img, cell_w, cell_h)
        canvas[y : y + cell_h, x : x + cell_w] = cell
        cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), (200, 200, 200), 1)

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        label = scene_label(float(np.mean(gray)))
        cv2.putText(
            canvas,
            label,
            (x + 8, y + cell_h + 20),
            font,
            0.52,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render D2-City sample montage")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "config" / "d2city_eval.yaml"),
    )
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR.parent / "results" / "figures" / "d2city_dataset_samples.png"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = get_d2city_processed_dir(cfg)
    split = cfg["d2city"]["split"]
    city_id = cfg["d2city"]["city_id"]
    frames_dir = processed / split / "frames" / city_id
    videos_dir = processed / split / "videos" / city_id

    items: list[tuple[np.ndarray, str]] = []
    if videos_dir.exists():
        items = pick_from_videos(videos_dir, n=args.n)
    if len(items) < args.n and frames_dir.exists():
        items = pick_diverse_frames(frames_dir, n=args.n)
    if not items:
        raise SystemExit(
            f"No frames found under {videos_dir} or {frames_dir}\n"
            "Run extract_d2city.sh and prepare_d2city_jsonl.py first."
        )

    montage = build_montage(items)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Wrote {out}")
    print(f"Panels: {len(items)}")


if __name__ == "__main__":
    main()
