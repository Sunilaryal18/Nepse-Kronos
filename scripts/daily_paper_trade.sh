#!/bin/zsh
# Weekday 5 pm job on this Mac: update NEPSE data, run the paper portfolio and rebuild the local copy of the
# report. Publishing is done by GitHub Actions (.github/workflows/daily-paper-trade.yml);
# this only pulls the published reports so the local copy stays in step.
# Scheduled by ~/Library/LaunchAgents/com.sunilaryal.nepse-kronos.daily.plist (see nepse_kronos/README.md section 8).
set -u
cd "${0:A:h}/.." || exit 1
LOG=outputs/nepse/daily.log
mkdir -p outputs/nepse
exec >>"$LOG" 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily paper-trading run (local)"

# Also start the GitHub job, because GitHub sometimes skips its own schedule. Extra runs on the same day publish nothing.
/opt/homebrew/bin/gh workflow run daily-paper-trade.yml -R Sunilaryal18/Nepse-Kronos && echo "started the GitHub job" ||
    echo "could not start the GitHub job (its own schedule still applies)"

if ! .venv311/bin/python -m nepse_kronos.paper_trade >/dev/null; then
    echo "paper_trade failed"
    exit 1
fi
.venv311/bin/python -m nepse_kronos.dashboard >/dev/null || echo "dashboard failed"
git pull -q --ff-only personal main && echo "pulled published reports" || echo "pull skipped (local changes or offline)"
