#!/bin/bash
# Pack or unpack the data the daily job keeps between runs, encrypted with $NEPSE_DATA_KEY.
#   scripts/data_bundle.sh pack   nepse-data.tar.gz.enc
#   scripts/data_bundle.sh unpack nepse-data.tar.gz.enc
# The encrypted file is stored as a release asset on GitHub. Without the key it can't be read, so the
# collected ShareSansar/MeroLagani/NEPSE data isn't republished. Prices that are re-downloaded every run
# (the community dataset clones, raw/clean prices) are left out, and so are your real holdings.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NEPSE_DATA_KEY:?set NEPSE_DATA_KEY}"

PATHS=(data/nepse/meta data/nepse/live data/nepse/source/floorsheet_daily data/nepse/source/nrb
       data/nepse/source/nepse-history data/nepse/source/nepse_api data/nepse/source/sharesansar
       data/nepse/source/merolagani outputs/nepse/paper outputs/nepse/picks_track_record.csv)

case "${1:-}" in
  pack)
    existing=(); for p in "${PATHS[@]}"; do [ -e "$p" ] && existing+=("$p"); done
    # COPYFILE_DISABLE stops macOS tar adding ._ metadata files, which the job would try to read as data
    COPYFILE_DISABLE=1 tar -czf - --exclude='.git' --exclude='._*' "${existing[@]}" |
      openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -pass env:NEPSE_DATA_KEY -out "$2"
    echo "packed ${#existing[@]} paths into $2 ($(du -h "$2" | cut -f1))" ;;
  unpack)
    openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:NEPSE_DATA_KEY -in "$2" | tar -xzf -
    echo "unpacked $2" ;;
  *) echo "usage: $0 pack|unpack FILE" >&2; exit 2 ;;
esac
