#!/usr/bin/env python3
"""Smoke-test the deployed LocateAnything Modal API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paths import (  # noqa: E402
    get_d2city_processed_dir,
    load_config,
    resolve_under_data_root,
)


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def resolve_image_from_jsonl(config_path: Path) -> Path:
    cfg = load_config(config_path)
    jsonl_path = resolve_under_data_root(cfg, cfg["output"]["jsonl"])
    image_root = get_d2city_processed_dir(cfg)
    with open(jsonl_path, encoding="utf-8") as f:
        first = json.loads(f.readline())
    image_path = image_root / first["image_path"]
    if not image_path.exists():
        raise FileNotFoundError(f"Frame from JSONL not found: {image_path}")
    return image_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Test LocateAnything Modal /detect")
    parser.add_argument(
        "--url",
        default=os.environ.get("MODAL_API_URL", ""),
        help="Modal base URL (or set MODAL_API_URL)",
    )
    parser.add_argument("--image", help="Path to a test image")
    parser.add_argument(
        "--from-jsonl",
        action="store_true",
        help="Use the first frame from config output JSONL (after prepare_d2city_jsonl.py)",
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "config" / "d2city_eval.yaml"),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["car", "bus", "truck", "person", "bicycle", "motorcycle"],
    )
    parser.add_argument("--generation-mode", default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if not args.url:
        print("ERROR: pass --url or set MODAL_API_URL", file=sys.stderr)
        return 1

    if args.from_jsonl:
        if args.image:
            print("ERROR: use either --image or --from-jsonl, not both", file=sys.stderr)
            return 1
        try:
            image_path = resolve_image_from_jsonl(Path(args.config))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            print(f"ERROR: --from-jsonl failed: {exc}", file=sys.stderr)
            return 1
        print(f"Using first JSONL frame: {image_path}")
    elif args.image:
        image_path = Path(args.image)
    else:
        print("ERROR: pass --image PATH or --from-jsonl", file=sys.stderr)
        return 1

    base_url = args.url.rstrip("/")
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    payload = {
        "image_base64": image_to_base64(image_path),
        "categories": args.categories,
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
    }

    response = requests.post(
        f"{base_url}/detect",
        json=payload,
        timeout=args.timeout,
    )
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)
        return 1

    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
