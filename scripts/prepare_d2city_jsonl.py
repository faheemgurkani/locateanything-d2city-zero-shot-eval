#!/usr/bin/env python3
"""
Build LocateAnything-compatible JSONL + extracted JPEG frames from D2-City val split.

Output JSONL schema (matches eagle/Embodied/evaluation/inference_grounding_ddp.py):
  {
    "image_path": "val/frames/0008/<hash>/000086.jpg",
    "categories": ["car", "bus", ...],
    "gt": {"car": [[x1,y1,x2,y2], ...], ...},
    "dataset_name": "D2City",
    "task_name": "common_object_detection"
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.d2city_annotation_parser import parse_d2city_xml  # noqa: E402
from paths import (  # noqa: E402
    get_d2city_processed_dir,
    load_config,
    resolve_under_data_root,
)


def build_gt_for_frame(
    frame_anns: list[dict],
    label_map: dict[str, str | None],
    query_classes: list[str],
    width: int,
    height: int,
) -> dict[str, list[list[float]]]:
    gt: dict[str, list[list[float]]] = {c: [] for c in query_classes}
    for ann in frame_anns:
        mapped = label_map.get(ann["label"])
        if mapped is None or mapped not in gt:
            continue
        x1 = max(0.0, min(float(ann["xtl"]), width))
        y1 = max(0.0, min(float(ann["ytl"]), height))
        x2 = max(0.0, min(float(ann["xbr"]), width))
        y2 = max(0.0, min(float(ann["ybr"]), height))
        if x2 > x1 and y2 > y1:
            gt[mapped].append([x1, y1, x2, y2])
    return {k: v for k, v in gt.items() if v}


def parse_meta(xml_path: Path) -> tuple[int, int]:
    tree = ET.parse(xml_path)
    meta = tree.getroot().find("meta")
    if meta is None:
        return 1920, 1080
    w = int(meta.findtext("width", "1920"))
    h = int(meta.findtext("height", "1080"))
    return w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare D2-City JSONL for LocateAnything eval")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "config" / "d2city_eval.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Count samples without writing frames")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    processed = get_d2city_processed_dir(cfg)
    split = cfg["d2city"]["split"]
    city_id = cfg["d2city"]["city_id"]

    ann_dir = processed / split / "annotations" / city_id
    vid_dir = processed / split / "videos" / city_id

    query_classes: list[str] = cfg["eval"]["query_classes"]
    label_map: dict[str, str | None] = cfg["eval"]["label_map"]
    stride: int = cfg["eval"]["frame_stride"]
    max_per_video = cfg["eval"].get("max_frames_per_video")
    min_gt = cfg["eval"].get("min_gt_boxes", 1)

    out_jsonl = resolve_under_data_root(cfg, cfg["output"]["jsonl"])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not ann_dir.exists():
        sys.exit(
            f"Annotations not found at {ann_dir}\n"
            f"Run: bash scripts/extract_d2city.sh {split}"
        )

    entries = []
    xml_files = sorted(ann_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files in {ann_dir}")

    for xml_path in xml_files:
        video_id = xml_path.stem
        video_path = vid_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"  SKIP (no video): {video_id}")
            continue

        width, height = parse_meta(xml_path)
        frame_anns = parse_d2city_xml(str(xml_path))
        candidate_frames = sorted(f for f in frame_anns if f % stride == 0)

        written = 0
        cap = None if args.dry_run else cv2.VideoCapture(str(video_path))

        for frame_idx in candidate_frames:
            gt = build_gt_for_frame(
                frame_anns[frame_idx], label_map, query_classes, width, height
            )
            n_boxes = sum(len(v) for v in gt.values())
            if n_boxes < min_gt:
                continue
            if max_per_video is not None and written >= max_per_video:
                break

            rel_image = f"{split}/frames/{city_id}/{video_id}/{frame_idx:06d}.jpg"
            entry = {
                "image_path": rel_image,
                "categories": query_classes,
                "gt": gt,
                "dataset_name": cfg["eval"]["dataset_name"],
                "task_name": cfg["eval"]["task_name"],
            }
            entries.append(entry)

            if not args.dry_run:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"  WARN: failed frame {frame_idx} in {video_id}")
                    entries.pop()
                    continue
                out_path = processed / rel_image
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1

        if cap is not None:
            cap.release()
        print(f"  {video_id}: {written} frames")

    if not args.dry_run:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nTotal eval samples: {len(entries)}")
    if args.dry_run:
        print("(dry-run — no frames or JSONL written)")
    else:
        print(f"JSONL:  {out_jsonl}")
        print(f"Frames: {processed / split / 'frames' / city_id}")


if __name__ == "__main__":
    main()
