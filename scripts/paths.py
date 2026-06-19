"""Resolve data and output paths from config/d2city_eval.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

ABLATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ABLATION_ROOT / "config" / "d2city_eval.yaml"


def load_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_ablation_root() -> Path:
    return ABLATION_ROOT


def get_data_root(cfg: dict) -> Path:
    """
    Where D2-City lives on disk.

    - monorepo (default): <parent-repo>/data  (e.g. ../../../data when nested)
    - local:              <ablation-root>/data  (standalone clone)
    - data_root override: explicit path (absolute or relative to ablation root)
    """
    paths = cfg.get("paths", {})
    override = paths.get("data_root")
    if override:
        p = Path(override)
        return p.resolve() if p.is_absolute() else (ABLATION_ROOT / p).resolve()

    mode = paths.get("data_root_mode", "monorepo")
    if mode == "local":
        return (ABLATION_ROOT / "data").resolve()

    repo_root = Path(paths.get("repo_root", cfg.get("repo_root", "../../..")))
    return (ABLATION_ROOT / repo_root / "data").resolve()


def get_d2city_raw_dir(cfg: dict) -> Path:
    sub = cfg["d2city"].get("raw_subdir", "d2_city")
    return get_data_root(cfg) / sub


def get_d2city_processed_dir(cfg: dict) -> Path:
    sub = cfg["d2city"].get("processed_subdir", "d2_city/processed")
    return get_data_root(cfg) / sub


def resolve_under_data_root(cfg: dict, rel_path: str) -> Path:
    """Resolve a path relative to data root (e.g. d2_city/processed/...)."""
    rel = rel_path.removeprefix("data/")
    return get_data_root(cfg) / rel


def get_modal_output_jsonl(cfg: dict) -> Path:
    rel = cfg.get("modal", {}).get(
        "output_jsonl",
        "d2_city/processed/_annotations/box_eval/D2City_val_modal_answer.jsonl",
    )
    return resolve_under_data_root(cfg, rel)


def get_results_dir(cfg: dict) -> Path:
    rel = cfg["output"].get("results_dir", "results/D2City_val")
    p = Path(rel)
    return p if p.is_absolute() else (ABLATION_ROOT / p).resolve()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Print resolved paths from config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "keys",
        nargs="*",
        choices=["data-root", "raw", "processed", "jsonl", "modal-jsonl", "results", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    keys = args.keys if args.keys else ["all"]

    mapping = {
        "data-root": get_data_root(cfg),
        "raw": get_d2city_raw_dir(cfg),
        "processed": get_d2city_processed_dir(cfg),
        "jsonl": resolve_under_data_root(cfg, cfg["output"]["jsonl"]),
        "modal-jsonl": get_modal_output_jsonl(cfg),
        "results": get_results_dir(cfg),
    }

    if "all" in keys:
        out = {k: str(v) for k, v in mapping.items()}
        print(json.dumps(out, indent=2))
    else:
        for k in keys:
            print(mapping[k])
