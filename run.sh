#!/usr/bin/env bash
set -euo pipefail
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export C10_LOG_LEVEL=${C10_LOG_LEVEL:-ERROR}
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
export CAFFE2_LOG_LEVEL=${CAFFE2_LOG_LEVEL:-ERROR}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}

INTERVAL="${PROCESS_INTERVAL:-60}"

python -u detect_and_crop_yolo.py
# Identify step should not crash the container if crops are empty
while true; do
  python -u detect_and_crop_yolo.py || true
  python -u identify_cats.py || true
  sleep "${INTERVAL}"
done