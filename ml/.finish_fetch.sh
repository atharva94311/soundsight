#!/usr/bin/env bash
# Waits for the four in-flight dev parts to land, then fetches whatever is left
# and extracts. Keys off the .done markers rather than process names, so it
# cannot match its own command line.
cd "$(dirname "$0")"
RAW=data/raw
need="FSD50K.dev_audio.z02 FSD50K.dev_audio.z03 FSD50K.dev_audio.z04 FSD50K.dev_audio.z05"
while true; do
  missing=0
  for f in $need; do [ -f "$RAW/$f.done" ] || missing=1; done
  [ "$missing" -eq 0 ] && break
  sleep 30
done
echo "=== in-flight parts complete, fetching remainder + extracting ==="
exec ./fetch_data.sh
