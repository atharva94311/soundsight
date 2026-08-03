#!/usr/bin/env bash
# Waits until every archive part has its .done marker, then runs fetch_data.sh,
# which skips the completed downloads and performs the merge + extract.
#
# Keyed on .done marker files rather than process names so it cannot match its
# own command line, and so it survives the download processes being restarted.
cd "$(dirname "$0")"
RAW=data/raw

PARTS="FSD50K.dev_audio.z01 FSD50K.dev_audio.z02 FSD50K.dev_audio.z03 \
FSD50K.dev_audio.z04 FSD50K.dev_audio.z05 FSD50K.dev_audio.zip \
FSD50K.eval_audio.z01 FSD50K.eval_audio.zip ESC-50-master.zip"

echo "waiting for $(echo $PARTS | wc -w | tr -d ' ') archive parts…"
while true; do
  missing=""
  for f in $PARTS; do
    [ -f "$RAW/$f.done" ] || missing="$missing $f"
  done
  [ -z "$missing" ] && break

  # If nothing is downloading and parts are still missing, a transfer died.
  # Restart the stalled ones rather than waiting forever.
  if [ "$(ps -eo comm | grep -c '^curl$')" -eq 0 ]; then
    echo "$(date '+%H:%M:%S') no transfers running, restarting:$missing"
    ZEN="https://zenodo.org/records/4060432/files"
    for f in $missing; do
      ( curl -fsSL -C - --retry 8 --retry-delay 5 --retry-all-errors \
          "$ZEN/$f?download=1" -o "$RAW/$f" && touch "$RAW/$f.done" ) &
    done
  fi
  sleep 60
done

echo "$(date '+%H:%M:%S') all parts present — merging and extracting"
./fetch_data.sh
