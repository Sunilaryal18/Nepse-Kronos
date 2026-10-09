#!/bin/zsh
# Weekday 5 pm job: update NEPSE data, run the paper portfolio, publish the day's report to the personal repo.
# Scheduled by ~/Library/LaunchAgents/com.sunilaryal.nepse-kronos.daily.plist (see nepse_kronos/README.md section 8).
set -u
cd "${0:A:h}/.." || exit 1
LOG=outputs/nepse/daily.log
mkdir -p outputs/nepse reports/paper
exec >>"$LOG" 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily paper-trading run"

if ! .venv311/bin/python -m nepse_kronos.paper_trade >/dev/null; then
    echo "paper_trade failed; nothing published"
    exit 1
fi

latest=$(ls -1 outputs/nepse/paper/reports/*.md | sort | tail -1)
cp "$latest" reports/paper/
cp "$latest" reports/paper/LATEST.md
cp outputs/nepse/paper/trades.csv reports/paper/trades.csv

git add reports/paper
if git diff --cached --quiet -- reports/paper; then
    echo "no new trading day; nothing to publish"
    exit 0
fi
git commit -q -m "Daily paper report $(basename "$latest" .md)" -- reports/paper
git push -q personal HEAD:main && echo "published $(basename "$latest")" || echo "push failed (report committed locally)"
