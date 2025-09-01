#!/usr/bin/env bash
set -euo pipefail
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore::UserWarning}

python -u detect_and_crop_yolo.py
# Identify step should not crash the container if crops are empty
if [[ -f identify_cats_nn.py ]]; then
  python -u identify_cats_nn.py || true
fi
