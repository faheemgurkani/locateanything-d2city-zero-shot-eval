#!/usr/bin/env bash
# Regenerate all article / reproducibility figures.
# Requires: metrics JSON + modal answer JSONL (run reproduce_results.sh first).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3}"

echo "=== Rendering figures ==="
"${PYTHON}" scripts/render_paper_benchmark_chart.py
"${PYTHON}" scripts/render_d2city_montage.py
"${PYTHON}" scripts/render_detection_figure.py
"${PYTHON}" scripts/render_iou_curve_chart.py
"${PYTHON}" scripts/render_gt_pred_comparison.py
"${PYTHON}" scripts/render_latency_chart.py

echo ""
echo "=== Figures written to results/figures/ ==="
ls -1 results/figures/*.{png,svg} 2>/dev/null || ls -1 results/figures/*.png
