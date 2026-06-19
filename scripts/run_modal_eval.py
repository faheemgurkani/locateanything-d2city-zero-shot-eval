#!/usr/bin/env python3
"""
Batch D2-City eval via Modal /detect endpoint.

Reads JSONL from prepare_d2city_jsonl.py, POSTs each frame to Modal,
writes answer.jsonl compatible with eagle/Embodied/evaluation/metrics/other_metric.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paths import (  # noqa: E402
    get_d2city_processed_dir,
    get_modal_output_jsonl,
    load_config,
    resolve_under_data_root,
)


def parse_extracted_predictions(answer: str, width: int, height: int) -> dict[str, list[list[float]]]:
    """Parse <ref>cat</ref><box>...</box> into per-category xyxy boxes (pixels)."""
    result: dict[str, list[list[float]]] = {}
    ref_pattern = r"<ref>([^<]+)</ref>((?:<box>.*?</box>)+)"
    box_pattern = (
        r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
    )

    for category, boxes_str in re.findall(ref_pattern, answer):
        cat = category.strip().lower()
        if cat not in result:
            result[cat] = []
        for match in re.findall(box_pattern, boxes_str):
            x1, y1, x2, y2 = (float(v) for v in match)
            result[cat].append(
                [
                    x1 / 1000 * width,
                    y1 / 1000 * height,
                    x2 / 1000 * width,
                    y2 / 1000 * height,
                ]
            )
    return result


def call_detect(
    base_url: str,
    image_path: Path,
    categories: list[str],
    generation_mode: str,
    max_new_tokens: int,
    timeout: int,
) -> dict:
    payload = {
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "categories": categories,
        "generation_mode": generation_mode,
        "max_new_tokens": max_new_tokens,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/detect",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="D2-City batch eval via Modal API")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "config" / "d2city_eval.yaml"),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MODAL_API_URL", ""),
        help="Modal base URL (or set MODAL_API_URL)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max samples (debug)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    modal_cfg = cfg.get("modal", {})
    base_url = args.url or modal_cfg.get("api_url", "")
    if not base_url and not args.dry_run:
        sys.exit("ERROR: pass --url or set MODAL_API_URL / config.modal.api_url")

    image_root = get_d2city_processed_dir(cfg)
    input_jsonl = resolve_under_data_root(cfg, cfg["output"]["jsonl"])
    output_jsonl = get_modal_output_jsonl(cfg)

    generation_mode = modal_cfg.get("generation_mode", cfg["model"]["generation_mode"])
    max_new_tokens = modal_cfg.get("max_new_tokens", cfg["inference"]["max_new_tokens"])
    timeout = modal_cfg.get("timeout_sec", 600)

    entries = []
    with open(input_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if args.limit:
        entries = entries[: args.limit]

    print(f"Samples: {len(entries)}")
    print(f"Modal URL: {base_url or '(dry-run)'}")

    if args.dry_run:
        return

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for entry in tqdm(entries, desc="Modal detect"):
        rel_path = entry["image_path"]
        full_path = image_root / rel_path
        if not full_path.exists():
            tqdm.write(f"SKIP missing image: {full_path}")
            continue

        try:
            resp = call_detect(
                base_url,
                full_path,
                entry["categories"],
                generation_mode,
                max_new_tokens,
                timeout,
            )
        except requests.RequestException as exc:
            tqdm.write(f"ERROR {rel_path}: {exc}")
            continue

        answer = resp.get("answer", "")
        w, h = resp.get("image_size", [0, 0])
        extracted = parse_extracted_predictions(answer, w, h)

        categories_str = "</c>".join(entry["categories"])
        question = (
            "Locate all the instances that matches the following description: "
            f"{categories_str}."
        )

        results.append(
            {
                "image_path": rel_path,
                "extracted_predictions": extracted,
                "gt": entry["gt"],
                "question": question,
                "dataset_name": entry["dataset_name"],
                "raw_response": answer,
                "task_name": entry["task_name"],
                "latency_ms": resp.get("latency_ms"),
            }
        )

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} predictions → {output_jsonl}")


if __name__ == "__main__":
    main()
