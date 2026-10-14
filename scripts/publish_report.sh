#!/bin/bash
# Copy the day's HTML report (with the real holdings), trades and settings into reports/paper/ (what gets published).
# index.html is always the latest report; <date>.html keeps each day. Prints the report's date.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/paper
day=$(python3 -c "import json; print(json.load(open('outputs/nepse/paper/state.json'))['date'])")
cp outputs/nepse/dashboard.html "reports/paper/$day.html"
cp outputs/nepse/dashboard.html reports/paper/index.html
cp outputs/nepse/paper/trades.csv outputs/nepse/paper/config.json reports/paper/
echo "$day"
