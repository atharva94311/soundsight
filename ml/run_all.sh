#!/usr/bin/env bash
# Fetch -> cache -> train -> export -> verify.
#
# Every stage is resumable and skips work it has already done, so re-running
# after an interruption or a code change is cheap. Safe to run repeatedly.
set -euo pipefail

cd "$(dirname "$0")"
PY=.venv/bin/python
EPOCHS="${EPOCHS:-40}"

if [[ ! -x "$PY" ]]; then
  echo "== creating venv =="
  uv venv --python 3.12 .venv
  uv pip install --python "$PY" torch numpy soundfile scipy tqdm
fi

echo
echo "== 1/5  datasets =="
./fetch_data.sh

echo
echo "== 2/5  window cache =="
$PY -m vas_ml.datasets

echo
echo "== 3/5  train ($EPOCHS epochs) =="
$PY train.py --epochs "$EPOCHS"

echo
echo "== 4/5  export to browser + firmware =="
$PY export.py

echo
echo "== 5/5  parity checks =="
$PY -m tests.test_parity
$PY -m tests.test_inference_parity
$PY -m tests.test_c_parity

echo
echo "Done. Serve the twin and click 'Listen with my mic':"
echo "    python3 -m http.server 8123 --directory ../vas3d"
