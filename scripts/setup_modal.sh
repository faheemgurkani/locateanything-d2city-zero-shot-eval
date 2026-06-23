#!/usr/bin/env bash
# Install Modal client (uses project .venv, or parent repo .venv as fallback).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ABLATION_ROOT}/../.." && pwd)"

if [[ -f "${ABLATION_ROOT}/.venv/bin/activate" ]]; then
  VENV="${ABLATION_ROOT}/.venv"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  VENV="${REPO_ROOT}/.venv"
else
  echo "ERROR: no .venv found. Run: bash scripts/setup_env.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip
pip install "modal>=0.73.0" requests pyyaml tqdm pycocotools shapely matplotlib

echo ""
echo "Modal client installed in ${VENV}."
echo "Authenticate: modal token set ..."
echo ""
echo "Next (from project root):"
echo "  python -m modal run modal/download.py::download_model"
echo "  python -m modal deploy modal/app.py"
echo "  export MODAL_API_URL=https://YOUR-WORKSPACE--locateanything-3b-....modal.run"
