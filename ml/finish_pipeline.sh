#!/usr/bin/env bash
# Everything remaining, unattended: wait for the downloads, extract, rebuild the
# window cache with all six classes, train, export to both targets, and verify.
#
# Each stage prints a banner so the log is readable, and the run stops at the
# first failure rather than reporting success on a half-built model.
set -o pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
EPOCHS="${EPOCHS:-40}"
LOG() { echo; echo "########## $(date '+%H:%M:%S')  $*"; echo; }

LOG "1/6  waiting for downloads, then extracting"
./extract_when_ready.sh || { echo "FAILED: download/extract"; exit 1; }

LOG "2/6  rebuilding window cache (auto-detects the new FSD50K audio)"
$PY -m vas_ml.datasets || { echo "FAILED: cache"; exit 1; }

LOG "3/6  training ($EPOCHS epochs)"
$PY train.py --epochs "$EPOCHS" || { echo "FAILED: train"; exit 1; }

LOG "4/6  exporting to browser + firmware"
$PY export.py || { echo "FAILED: export"; exit 1; }

LOG "5/6  parity checks"
rc=0
$PY -m tests.test_parity           || rc=1
$PY -m tests.test_inference_parity || rc=1
$PY -m tests.test_c_parity         || rc=1
[ $rc -eq 0 ] || { echo "FAILED: parity"; exit 1; }

LOG "6/6  done"
echo "=== dataset ==="
$PY -c "
import json,pathlib
m=json.loads(pathlib.Path('data/cache/meta.json').read_text())
for s,v in m['splits'].items():
    print(f'  {s:6s} {v[\"windows\"]:6d} windows from {v[\"clips\"]:5d} clips  {v[\"per_class\"]}')
"
echo "=== test metrics ==="
$PY -c "
import json,pathlib
m=json.loads(pathlib.Path('artifacts/metrics.json').read_text())
print(f'  accuracy {m[\"accuracy\"]:.3f}    false-alarm rate {m[\"false_alarm_rate\"]:.4f}')
for c,v in m['per_class'].items():
    print(f'  {c:12s} recall {v[\"recall\"]:.3f}  precision {v[\"precision\"]:.3f}  n={v[\"n\"]}')
"
echo
echo "PIPELINE COMPLETE"
