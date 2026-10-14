#!/bin/bash
# One-time setup of the GitHub Actions daily job (run from your Mac after `gh auth login` with your personal account).
#   1. makes an encryption key (kept in ~/.config/nepse-kronos/data_key) and stores it as the NEPSE_DATA_KEY secret
#   2. packs the data from this Mac, encrypts it and uploads it to the "data-store" release
#   3. turns on GitHub Pages (the daily report's web address) and starts the first run
# Re-running it is safe: the key is reused and the stored data is replaced with this Mac's copy.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=Sunilaryal18/Nepse-Kronos
KEY_FILE="$HOME/.config/nepse-kronos/data_key"

if [ ! -f "$KEY_FILE" ]; then
    mkdir -p "$(dirname "$KEY_FILE")"
    (umask 077; openssl rand -base64 48 > "$KEY_FILE")
    echo "Made a new key in $KEY_FILE (keep it: without it the stored data can't be read)"
fi
gh secret set NEPSE_DATA_KEY -R "$REPO" < "$KEY_FILE"

bundle="$(mktemp -d)/nepse-data.tar.gz.enc"
NEPSE_DATA_KEY="$(cat "$KEY_FILE")" scripts/data_bundle.sh pack "$bundle"
gh release view data-store -R "$REPO" >/dev/null 2>&1 ||
    gh release create data-store -R "$REPO" --prerelease --title "Data store for the daily job" \
        --notes "Encrypted working data for the daily paper-trading job. Not a software release; it can't be read without the key."
gh release upload data-store "$bundle" -R "$REPO" --clobber
rm -f "$bundle"

gh api -X POST "repos/$REPO/pages" -f build_type=workflow >/dev/null 2>&1 ||
    gh api -X PUT "repos/$REPO/pages" -f build_type=workflow >/dev/null
gh workflow run daily-paper-trade.yml -R "$REPO"
echo "First run started: https://github.com/$REPO/actions"
echo "Daily report (after the run): https://sunilaryal18.github.io/Nepse-Kronos/"
