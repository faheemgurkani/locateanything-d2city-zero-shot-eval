#!/usr/bin/env bash
# One-time environment setup for LocateAnything zero-shot D2-City eval.
# Does NOT download the model or run inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EMBODIED="${ABLATION_ROOT}/eagle/Embodied"
VENV="${ABLATION_ROOT}/.venv"

echo "=== LocateAnything ablation env setup ==="
echo "Ablation root: ${ABLATION_ROOT}"
echo "Embodied:      ${EMBODIED}"

# --- Python venv ---
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- LocateAnything package (Eagle/Embodied) ---
pip install -e "${EMBODIED}"

# --- FastEvaluate for COCO/LVIS-style mAP (optional but installed) ---
FASTEVAL="${EMBODIED}/evaluation/fastevaluate"
if [[ -d "${FASTEVAL}" ]]; then
  pip install -e "${FASTEVAL}"
fi
pip install shapely pyyaml tqdm opencv-python-headless

# --- Flash Attention 2 (recommended for la_flash / vision_attn) ---
# Requires CUDA. Skip if no GPU driver on this machine.
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  pip install flash-attn --no-build-isolation || echo "WARN: flash-attn install failed; use attn=sdpa fallback"
else
  echo "WARN: CUDA not available here — install flash-attn on the GPU machine later."
fi

# --- Hugging Face CLI for model download ---
pip install "huggingface_hub[cli]"

echo ""
echo "=== Setup complete ==="
echo "Activate:  source ${VENV}/bin/activate"
echo "Next:      bash scripts/extract_d2city.sh val"
echo "Then:      python scripts/prepare_d2city_jsonl.py"
echo "Model:     hf download nvidia/LocateAnything-3B --local-dir models/LocateAnything-3B"
echo "           (accept NVIDIA model license on Hugging Face first)"
