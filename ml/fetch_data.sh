#!/usr/bin/env bash
# Downloads the public datasets the classifier trains on.
# Resumable: re-run it after an interrupted download and curl -C - picks up where it stopped.
set -euo pipefail

DATA="$(cd "$(dirname "$0")" && pwd)/data"
RAW="$DATA/raw"
mkdir -p "$RAW"

ZEN="https://zenodo.org/records/4060432/files"

get() {  # get <url> <dest>
  local url="$1" dest="$2"
  if [[ -f "$dest.done" ]]; then echo "  ✓ $(basename "$dest") (cached)"; return; fi
  echo "  ↓ $(basename "$dest")"
  curl -fsSL -C - --retry 5 --retry-delay 5 --retry-all-errors "$url" -o "$dest"
  touch "$dest.done"
}

# Zenodo throttles each connection rather than the account, so a single
# sequential fetch of 31 GB runs at ~1.5 MB/s while four in parallel run at
# ~1.5 MB/s *each*. Downloads are resumable (curl -C -), so an interrupted run
# picks up mid-file rather than starting the part over.
PARALLEL="${PARALLEL:-4}"

# Batched rather than a rolling pool: macOS still ships bash 3.2, which has no
# `wait -n`. Waiting for a whole batch wastes a little tail time per batch and
# needs no bash 4+ features.
get_all() {  # get_all <url|dest pairs on stdin>
  local pids="" url dest rc=0 running=0
  while read -r url dest; do
    [[ -z "$url" ]] && continue
    if [[ -f "$dest.done" ]]; then echo "  ✓ $(basename "$dest") (cached)"; continue; fi
    echo "  ↓ $(basename "$dest")"
    ( curl -fsSL -C - --retry 5 --retry-delay 5 --retry-all-errors "$url" -o "$dest" \
      && touch "$dest.done" ) &
    pids="$pids $!"
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then
      for p in $pids; do wait "$p" || rc=1; done
      pids=""; running=0
    fi
  done
  for p in $pids; do wait "$p" || rc=1; done
  return $rc
}

echo "== FSD50K ground truth + metadata =="
get "$ZEN/FSD50K.ground_truth.zip?download=1" "$RAW/FSD50K.ground_truth.zip"
get "$ZEN/FSD50K.metadata.zip?download=1"     "$RAW/FSD50K.metadata.zip"

echo "== FSD50K audio (31 GB, $PARALLEL at a time) =="
get_all <<EOF
$ZEN/FSD50K.dev_audio.z01?download=1  $RAW/FSD50K.dev_audio.z01
$ZEN/FSD50K.dev_audio.z02?download=1  $RAW/FSD50K.dev_audio.z02
$ZEN/FSD50K.dev_audio.z03?download=1  $RAW/FSD50K.dev_audio.z03
$ZEN/FSD50K.dev_audio.z04?download=1  $RAW/FSD50K.dev_audio.z04
$ZEN/FSD50K.dev_audio.z05?download=1  $RAW/FSD50K.dev_audio.z05
$ZEN/FSD50K.dev_audio.zip?download=1  $RAW/FSD50K.dev_audio.zip
$ZEN/FSD50K.eval_audio.z01?download=1 $RAW/FSD50K.eval_audio.z01
$ZEN/FSD50K.eval_audio.zip?download=1 $RAW/FSD50K.eval_audio.zip
EOF

echo "== ESC-50 (600 MB) =="
get "https://github.com/karoldvl/ESC-50/archive/master.zip" "$RAW/ESC-50-master.zip"

echo
echo "== extracting =="
cd "$RAW"

unz() {  # unz <marker-dir> <zip>
  if [[ -d "$DATA/$1" ]]; then echo "  ✓ $1 (already extracted)"; return; fi
  echo "  ⇥ $2 → $1"
  unzip -q -o "$2" -d "$DATA"
}

unz FSD50K.ground_truth FSD50K.ground_truth.zip
unz FSD50K.metadata     FSD50K.metadata.zip
unz ESC-50-master       ESC-50-master.zip

# Split archives must be concatenated into one zip before extraction.
# `zip -s 0` does that; it needs the final .zip plus every .zNN part alongside it.
for set in dev eval; do
  if [[ -d "$DATA/FSD50K.${set}_audio" ]]; then
    echo "  ✓ FSD50K.${set}_audio (already extracted)"
    continue
  fi
  if [[ ! -f "$RAW/${set}_unsplit.zip" ]]; then
    echo "  ⊕ merging FSD50K.${set}_audio parts"
    zip -q -s 0 "FSD50K.${set}_audio.zip" --out "${set}_unsplit.zip"
  fi
  echo "  ⇥ FSD50K.${set}_audio"
  unzip -q -o "${set}_unsplit.zip" -d "$DATA"
  rm -f "${set}_unsplit.zip"   # ~25 GB of scratch, drop it as soon as it is spent
done

echo
echo "done. clips:"
for d in FSD50K.dev_audio FSD50K.eval_audio; do
  printf '  %-22s %s\n' "$d" "$(ls "$DATA/$d" 2>/dev/null | wc -l | tr -d ' ')"
done
printf '  %-22s %s\n' "ESC-50" "$(ls "$DATA/ESC-50-master/audio" 2>/dev/null | wc -l | tr -d ' ')"
