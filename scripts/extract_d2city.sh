#!/usr/bin/env bash
# Extract D2-City archives into a flat processed layout.
# Respects paths.data_root_mode in config/d2city_eval.yaml.
# Usage: bash scripts/extract_d2city.sh [val|train|test|all]
set -euo pipefail

SPLIT="${1:-val}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

DATA_ROOT="$("${PYTHON}" "${SCRIPT_DIR}/paths.py" data-root)"
RAW="${DATA_ROOT}/d2_city"
OUT="${DATA_ROOT}/d2_city/processed"

echo "Data root: ${DATA_ROOT}"
echo "Raw zips:  ${RAW}"
echo "Processed: ${OUT}"

extract_val() {
  mkdir -p "${OUT}/val/annotations" "${OUT}/val/videos"

  echo "=== Extracting validation annotations ==="
  unzip -qo "${RAW}/validation-annotation.zip" -d "${OUT}/val/annotations"

  echo "=== Extracting validation videos (nested zip) ==="
  mkdir -p "${OUT}/val/videos/_tmp"
  unzip -qo "${RAW}/validation-video.zip" -d "${OUT}/val/videos/_tmp"
  for nested in "${OUT}/val/videos/_tmp"/*.zip; do
    city="$(basename "${nested}" .zip)"
    mkdir -p "${OUT}/val/videos/${city}"
    unzip -qo "${nested}" -d "${OUT}/val/videos/${city}"
    if [[ -d "${OUT}/val/videos/${city}/${city}" ]]; then
      mv "${OUT}/val/videos/${city}/${city}"/*.mp4 "${OUT}/val/videos/${city}/" 2>/dev/null || true
      rmdir "${OUT}/val/videos/${city}/${city}" 2>/dev/null || true
    fi
  done
  rm -rf "${OUT}/val/videos/_tmp"
  echo "Val videos: $(find "${OUT}/val/videos" -name '*.mp4' | wc -l | tr -d ' ') MP4 files"
  echo "Val XML:    $(find "${OUT}/val/annotations" -name '*.xml' | wc -l | tr -d ' ') XML files"
}

extract_train() {
  mkdir -p "${OUT}/train/annotations" "${OUT}/train/videos"

  echo "=== Extracting training annotations ==="
  unzip -qo "${RAW}/training-annotation.zip" -d "${OUT}/train/annotations"

  echo "=== Extracting training videos (nested zips, ~8 GB) ==="
  mkdir -p "${OUT}/train/videos/_tmp"
  unzip -qo "${RAW}/training-video.zip" -d "${OUT}/train/videos/_tmp"
  for nested in "${OUT}/train/videos/_tmp"/*.zip; do
    city="$(basename "${nested}" .zip)"
    mkdir -p "${OUT}/train/videos/${city}"
    unzip -qo "${nested}" -d "${OUT}/train/videos/${city}"
    if [[ -d "${OUT}/train/videos/${city}/${city}" ]]; then
      mv "${OUT}/train/videos/${city}/${city}"/*.mp4 "${OUT}/train/videos/${city}/" 2>/dev/null || true
      rmdir "${OUT}/train/videos/${city}/${city}" 2>/dev/null || true
    fi
  done
  rm -rf "${OUT}/train/videos/_tmp"
}

extract_test() {
  mkdir -p "${OUT}/test/videos"

  echo "=== Extracting test videos (no annotations) ==="
  mkdir -p "${OUT}/test/videos/_tmp"
  unzip -qo "${RAW}/test-video.zip" -d "${OUT}/test/videos/_tmp"
  for nested in "${OUT}/test/videos/_tmp"/*.zip; do
    city="$(basename "${nested}" .zip)"
    mkdir -p "${OUT}/test/videos/${city}"
    unzip -qo "${nested}" -d "${OUT}/test/videos/${city}"
    if [[ -d "${OUT}/test/videos/${city}/${city}" ]]; then
      mv "${OUT}/test/videos/${city}/${city}"/*.mp4 "${OUT}/test/videos/${city}/" 2>/dev/null || true
      rmdir "${OUT}/test/videos/${city}/${city}" 2>/dev/null || true
    fi
  done
  rm -rf "${OUT}/test/videos/_tmp"
}

if [[ ! -d "${RAW}" ]]; then
  echo "ERROR: D2-City raw directory not found: ${RAW}"
  echo "See data/README.md for download and layout instructions."
  exit 1
fi

mkdir -p "${OUT}"

case "${SPLIT}" in
  val)   extract_val ;;
  train) extract_train ;;
  test)  extract_test ;;
  all)   extract_val; extract_train; extract_test ;;
  *)     echo "Usage: $0 [val|train|test|all]"; exit 1 ;;
esac

echo "=== Done. Processed data at: ${OUT} ==="
