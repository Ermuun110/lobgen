#!/usr/bin/env bash
# Convert + validate every filtered day for each ticker. Sequential: a liquid stock-day
# needs ~3 GB RAM. Usage: scripts/convert_all.sh [TICKER ...]   (default GOOG INTC)
set -euo pipefail
cd "$(dirname "$0")/.."
tickers=("${@:-GOOG INTC}")
read -r -a tickers <<< "${tickers[*]}"
failed=0
for f in data/filtered/20??-??-??.itch.gz; do
  date=$(basename "$f" .itch.gz)
  for t in "${tickers[@]}"; do
    npz="data/lobster/${t}_${date}_34200000_57600000_10.npz"
    if [[ ! -f "$npz" ]]; then
      echo "[$(date +%T)] convert $t $date"
      uv run lobgen convert "$f" --ticker "$t" --date "$date" > /dev/null
    fi
    echo "[$(date +%T)] validate $t $date"
    if ! uv run lobgen validate "$npz" > /dev/null; then
      echo "  VALIDATION FAILED: ${npz%.npz}.validate.json"
      failed=1
    fi
  done
done
echo "[$(date +%T)] all done (failed=$failed)"
exit $failed
