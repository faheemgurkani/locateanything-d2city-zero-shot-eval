"""One-time model weight download for LocateAnything-3B into a Modal Volume.

Volume is created lazily via ``Volume.from_name(..., create_if_missing=True)``.
Run from ablation root::

    python -m modal run modal/download.py::download_model

Requires Modal secret ``huggingface-secret`` with key ``HF_TOKEN``.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "main"
VOLUME_NAME = "locateanything-weights"
MODEL_DIR = Path("/models") / MODEL_ID

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("locateanything-download")


@app.function(
    image=download_image,
    volumes={MODEL_DIR.as_posix(): volume},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def download_model() -> str:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        token=os.environ["HF_TOKEN"],
    )
    volume.commit()
    return f"Model downloaded to {MODEL_DIR}"
