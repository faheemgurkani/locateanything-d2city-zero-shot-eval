#!/usr/bin/env bash
# End-to-end reproduction of D2-City zero-shot LocateAnything eval (Modal path).
#
# Usage:
#   bash scripts/reproduce_results.sh              # data + inference + metrics + figures
#   bash scripts/reproduce_results.sh --skip-modal # metrics + figures only (predictions exist)
#   bash scripts/reproduce_results.sh --data-only  # extract + prepare JSONL only
#
# Prerequisites: see README.md § Full replication guide
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3}"
SKIP_MODAL=0
DATA_ONLY=0
SKIP_FIGURES=0

for arg in "$@"; do
  case "${arg}" in
    --skip-modal) SKIP_MODAL=1 ;;
    --data-only)  DATA_ONLY=1 ;;
    --skip-figures) SKIP_FIGURES=1 ;;
    *) echo "Unknown flag: ${arg}"; exit 1 ;;
  esac
done

# shellcheck disable=SC1091
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  source "${ROOT}/.venv/bin/activate"
fi

step() { echo ""; echo "=== $* ==="; }

step "1/5 Verify paths"
"${PYTHON}" scripts/paths.py all

step "2/5 Extract D2-City validation split"
bash scripts/extract_d2city.sh val

step "3/5 Build eval JSONL + JPEG frames (~500 samples)"
"${PYTHON}" scripts/prepare_d2city_jsonl.py

if [[ "${DATA_ONLY}" -eq 1 ]]; then
  echo "Done (--data-only)."
  exit 0
fi

if [[ "${SKIP_MODAL}" -eq 0 ]]; then
  step "4/5 Modal batch inference"
  if [[ -z "${MODAL_API_URL:-}" ]]; then
    if [[ -f .env ]]; then
      # shellcheck disable=SC1091
      source .env
    fi
  fi
  if [[ -z "${MODAL_API_URL:-}" ]]; then
    echo "ERROR: set MODAL_API_URL (export or .env) after: python -m modal deploy modal/app.py"
    echo "See modal/README.md for one-time Modal setup."
    exit 1
  fi
  "${PYTHON}" scripts/run_modal_eval.py --url "${MODAL_API_URL}"
else
  step "4/5 Skipping Modal inference (--skip-modal)"
fi

step "5/5 Compute metrics"
mkdir -p results/D2City_val/modal
if [[ ! -d eagle/Embodied ]]; then
  echo "ERROR: clone Eagle first: git clone https://github.com/NVlabs/Eagle.git eagle"
  exit 1
fi
"${PYTHON}" eagle/Embodied/evaluation/metrics/other_metric.py \
  --data_path "$("${PYTHON}" scripts/paths.py modal-jsonl)" \
  --output_path results/D2City_val/modal/eval_results.json

if [[ "${SKIP_FIGURES}" -eq 0 ]]; then
  step "Bonus: render figures"
  bash scripts/render_all_figures.sh
fi

echo ""
echo "=== Reproduction complete ==="
echo "Metrics:  results/D2City_val/modal/eval_results.json"
echo "Predictions: $("${PYTHON}" scripts/paths.py modal-jsonl)"
echo "Figures:  results/figures/"
