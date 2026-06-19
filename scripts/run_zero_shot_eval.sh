#!/usr/bin/env bash
# Zero-shot LocateAnything inference + metrics on D2-City val.
# Prerequisites: setup_env.sh, extract_d2city.sh, prepare_d2city_jsonl.py, model download.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EMBODIED="${ABLATION_ROOT}/eagle/Embodied"
PYTHON="${PYTHON:-python3}"

resolve() {
  "${PYTHON}" "${SCRIPT_DIR}/paths.py" "$1"
}

MODEL_PATH="${MODEL_PATH:-${ABLATION_ROOT}/models/LocateAnything-3B}"
IMAGE_ROOT="${IMAGE_ROOT:-$(resolve processed)}"
TEST_JSONL="${TEST_JSONL:-$(resolve jsonl)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ABLATION_ROOT}/results/D2City_val/hybrid}"
GPUS="${GPUS:-1}"
GENERATION_MODE="${GENERATION_MODE:-hybrid}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Model not found at ${MODEL_PATH}"
  echo "Run: hf download nvidia/LocateAnything-3B --local-dir ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${TEST_JSONL}" ]]; then
  echo "ERROR: JSONL not found at ${TEST_JSONL}"
  echo "Run: python scripts/prepare_d2city_jsonl.py"
  exit 1
fi

if [[ ! -d "${EMBODIED}" ]]; then
  echo "ERROR: Eagle/Embodied not found at ${EMBODIED}"
  echo "Run: git clone https://github.com/NVlabs/Eagle.git ${ABLATION_ROOT}/eagle"
  echo "Then: bash scripts/setup_env.sh"
  exit 1
fi

# shellcheck disable=SC1091
if [[ -f "${ABLATION_ROOT}/.venv/bin/activate" ]]; then
  source "${ABLATION_ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${MODEL_PATH}:${EMBODIED}:${PYTHONPATH:-}"
export LA_FLASH_ATTN="${LA_FLASH_ATTN:-la_flash}"

mkdir -p "${OUTPUT_DIR}"
SAVE_PATH="${OUTPUT_DIR}/answer.jsonl"
EVAL_JSON="${OUTPUT_DIR}/eval_results.json"
LOG_FILE="${OUTPUT_DIR}/evaluation.log"

echo "=== Zero-shot D2-City eval ==="
echo "Model:     ${MODEL_PATH}"
echo "JSONL:     ${TEST_JSONL}"
echo "Images:    ${IMAGE_ROOT}"
echo "Output:    ${OUTPUT_DIR}"
echo "GPUs:      ${GPUS}"
echo "Mode:      ${GENERATION_MODE}"

cd "${EMBODIED}"

torchrun \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT:-29501}" \
  evaluation/inference_grounding_ddp.py \
  --world_size "${GPUS}" \
  --model_path "${MODEL_PATH}" \
  --test_jsonl_path "${TEST_JSONL}" \
  --image_root_dir "${IMAGE_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --max_new_tokens 8192 \
  --num_workers 4 \
  --eval_type box_eval \
  --generation_mode "${GENERATION_MODE}" \
  2>&1 | tee "${LOG_FILE}"

python evaluation/metrics/other_metric.py \
  --data_path "${SAVE_PATH}" \
  --output_path "${EVAL_JSON}"

python evaluation/metrics/analyze_speed.py \
  --log_file "${LOG_FILE}" \
  2>&1 | tee -a "${LOG_FILE}" || true

echo ""
echo "=== Done ==="
echo "Predictions: ${SAVE_PATH}"
echo "Metrics:     ${EVAL_JSON}"
echo "Log:         ${LOG_FILE}"
