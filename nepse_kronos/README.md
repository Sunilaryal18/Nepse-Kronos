# Kronos for NEPSE

Daily forecasts for Nepal Stock Exchange symbols using the Kronos foundation model.

```
nepse-open-data ──► fetch ──► data/nepse/raw ──► prepare ──► data/nepse/clean ──┬──► predict   (forecast + chart)
(public GitHub data)                                                         ├──► backtest  (accuracy vs. "no change")
                                                                             └──► finetune_csv (optional fine-tuning)
```

## 1. Setup

Use Python 3.11. The pinned `pandas==2.2.2` does not support Python 3.13 (it crashes in `pd.to_datetime`).

```bash
uv venv --python 3.11 .venv311
uv pip install --python .venv311/bin/python -r requirements.txt pytest pyyaml
source .venv311/bin/activate
```

Run every command below from the repository root.

## 2. Get data

```bash
python -m nepse_kronos.fetch
```

Downloads or updates [socrateai-official/nepse-open-data](https://github.com/socrateai-official/nepse-open-data) (MIT licence, updated daily) and writes one file per symbol to `data/nepse/raw/`:

- stocks keep their ticker: `NABIL.csv`, `NICA.csv`, ...
- indices get an `_INDEX` suffix: `NEPSE_INDEX.csv`, `BANKING_INDEX.csv`, ...

Stock prices are already adjusted for bonus and rights shares. For indices, the `volume` column is turnover in NPR, not shares.
You can also put your own CSV exports in `data/nepse/raw/`. Common column names such as `Date`, `Open`, `High`, `Low`, `LTP`, `Total Traded Quantity` and `Turnover` are recognised.

Use `--symbols NEPSE_INDEX NABIL` to write only some symbols, and `--no-sync` to skip the download.

## 3. Clean

```bash
python -m nepse_kronos.prepare
```

Writes `data/nepse/clean/<SYMBOL>.csv` in the format Kronos expects. Along the way it:

- fixes column names and numbers written with commas
- drops bad rows and duplicate dates
- drops the early close-only history of indices (before late 2016 they have no real candles), which otherwise makes Kronos forecast fake crashes

## 4. Forecast

```bash
python -m nepse_kronos.predict --symbol NABIL --pred-len 10
```

Writes `outputs/nepse/pred_NABIL.csv` and `.png`. Future dates follow the Monday–Friday NEPSE week, skipping the closures listed in `nepse_kronos/holidays.csv`. Keep that file up to date from NEPSE notices.

Useful options:

- `--lookback 128` uses a shorter context; it scored better than the default 400 on the index
- `--sample-count 50` (default) averages 50 sampled paths; fewer samples make results noticeably noisier
- `--device cpu` if the Apple GPU (`mps`) gives trouble

## 5. Check accuracy

```bash
python -m nepse_kronos.backtest --symbol NABIL --pred-len 5 --step 10 --lookback 128
```

Replays history: from every `--step`-th day it forecasts the next `--pred-len` days and compares with what happened. Add `--start-date YYYY-MM-DD` to score only later dates.

| metric | meaning | useful when |
|---|---|---|
| `mape` | average % error of the final forecast close | below `naive_mape` |
| `naive_mape` | same error for "price stays at the last close" | baseline |
| `direction_accuracy` | share of forecasts that got up/down right | above 0.5 |
| `rank_ic` | rank correlation of predicted vs actual moves | clearly above 0 |

### What we found (5-day horizon, October 2026)

| setup | windows | mape | naive | direction | rank_ic |
|---|---|---|---|---|---|
| pretrained, 2017–2026, lookback 400, 5 samples | 186 | 5.0% | 2.4% | 0.48 | −0.09 |
| pretrained, 2017–2026, lookback 128, 5 samples | 213 | 3.2% | 2.5% | 0.45 | −0.08 |
| pretrained, held-out year, lookback 128, 50 samples | 44 | 1.8% | 1.8% | 0.59 | 0.25 |
| fine-tuned, held-out year, lookback 128, 50 samples | 44 | 2.0% | 1.8% | 0.61 | 0.28 |
| NABIL (stock), pretrained, 2011–2026, lookback 128, 50 samples | 335 | 3.4% | 2.8% | 0.47 | 0.01 |

Kronos does **not** yet beat "no change" on error, for the index or for NABIL. Its positive direction and rank results on the latest year are encouraging but not statistically significant with 44 forecasts. Backtest a symbol before relying on its forecasts.

## 6. Fine-tune (optional)

```bash
python -c "
from huggingface_hub import snapshot_download as get
get('NeoQuasar/Kronos-Tokenizer-base', local_dir='pretrained/Kronos-Tokenizer-base')
get('NeoQuasar/Kronos-small', local_dir='pretrained/Kronos-small')
"
cd finetune_csv && python train_sequential.py --config configs/config_nepse_daily.yaml && cd ..
```

This takes about 35 minutes on a Mac CPU for NEPSE_INDEX. The training script uses CUDA if present, otherwise CPU; the Apple GPU is not used. The last 10% of the series is held out from training. Use the result with:

```bash
python -m nepse_kronos.predict --symbol NEPSE_INDEX \
    --model finetuned/NEPSE_daily/basemodel/best_model --tokenizer finetuned/NEPSE_daily/tokenizer/best_model
```

On the index, fine-tuning did not clearly help: it ranked moves about as well but picked up a downward bias. Compare with `backtest --start-date <held-out start>` before adopting a fine-tuned model.

## Tests

```bash
python -m pytest tests/nepse_pipeline -v
```

The tests use a stub predictor, so they never download model weights.

Forecasts are probabilistic research output, not investment advice.

## 7. Trading-system backtest

A full rules-based system: every trading day, check each holding and sell the next morning if a rule says so; fill empty slots with the best-ranked shares; refresh the ranking every 10 trading days. Real NEPSE costs (commission 0.24–0.36%, SEBON 0.015%, Rs 25 DP, slippage that grows with trade size, 10% capital-gains tax), sale money usable after 2 days, ±10% price limits, Rs 10 lakh capital, up to 20 shares.

```bash
python -m nepse_kronos.portfolio_backtest --signal lowvol --start 2017-06-01      # simple rules: seconds
python -m nepse_kronos.portfolio_backtest --signal kronos --start 2024-10-01 --sample-count 20   # ~1.5 h, resumable
```

Rules (fixed before testing): keep a share while it is in the top 50; sell if it closes 15% below its best close since buying; sell if held 60 days without a gain; don't re-buy within 20 days; less money in jumpy shares; no position above 10% of the share's normal daily trading.

Ranking signals: `momentum` (biggest 60-day rise), `steady` (rise ÷ jumpiness), `lowvol` (calmest shares), `kronos` (Kronos-small predicted 10-day return, 20 samples).

### Results (October 2026)

**2017-06 → 2026-10** (index **+61%**, worst drop −43%):

| Ranking | Agreed rules | Without loss/time limits | Fees + tax (agreed rules) | Picks right? (score, t) |
|---|---|---|---|---|
| momentum | −31% | −23% | Rs 8.5 lakh | 0.00 (0.1) |
| steady | −36% | −8% | Rs 8.2 lakh | 0.00 (0.0) |
| lowvol | −16% | **+50%** (worst drop −37%) | Rs 5.6 lakh | **0.08 (4.2)** |

**2024-10 → 2026-10** (index **−1.4%**, worst drop −17%):

| Ranking | Agreed rules | Without loss/time limits | Return with no costs | Top-20 still top-20 after 2 weeks | Picks right? (score, t) |
|---|---|---|---|---|---|
| kronos | **−34%** | −35% | −8% | 25% | 0.06 (2.9) |
| momentum | −16% | −20% | −1% | 62% | −0.01 (−0.2) |
| steady | −21% | −18% | — | — | −0.02 (−0.6) |
| lowvol | −9% | −6% | +3% | 81% | **0.15 (3.8)** |

### Verdict

- **Kronos ranking is not useful as a stock picker here.** It loses to the index and to every simple signal, in every year. Its picks are slightly better than chance on average (score 0.06, t = 2.9), but its top 20 reshuffles almost completely every two weeks, so the portfolio over-trades (1,074 trades, Rs 2.9 lakh in costs in two years) and even with zero costs it made −8%.
- **No version of the agreed rules beat simply holding the index.** Two things hurt most:
  1. the **15% loss limit** — NEPSE shares often fall that far and recover, so it sells near the bottom (removing it turns lowvol from −16% into +50% over 2017–2026);
  2. **costs** — fees and tax take roughly half of the gross gains.
- **The calmest shares (`lowvol`) are the one real signal found**: they reliably did better than other shares (score 0.08–0.15, strongly significant in both periods), and without the loss limit the portfolio came close to the index (+50% vs +61%) with smaller drops. Because that rule change was chosen *after* seeing the results, it needs confirming on future data before trusting it.

Caveats: opening-price fills with modelled slippage are an approximation; thinly traded shares may cost more. Cash dividends are not included for either the portfolio or the index.

## 8. Daily paper trading

Tests the most promising rule set from section 7 on **future** data, with pretend money:
calmest shares (top 20 of the 150 most-traded), keep while in the top 50, no automatic stop-loss, ranking refreshed every 10 trading days, real costs.

```bash
python -m nepse_kronos.paper_trade            # run each trading evening: downloads new prices, then reports
```

It prints (and saves to `outputs/nepse/paper/reports/<date>.md`):
- what the paper portfolio owns and its value vs the NEPSE index since the start,
- the orders for the next morning (what to buy or sell, and why).

The first run fixes the start date and settings in `outputs/nepse/paper/config.json`; delete that folder to start over.
Prices come from the community datasets plus, for days they haven't published yet, the NEPSE website itself
(`python -m nepse_kronos.live_prices`, run automatically by `paper_trade`: index history and per-share
high/low/close/volume for the ~200 most-traded shares and current holdings, ~3–4 minutes at 1 request/second).
NEPSE doesn't publish a share's opening price there, so for those newest days the open is taken as the previous
close; the community data replaces those days once it catches up. Run it after the market closes (3 pm).

## 9. "Top investor" upgrade

What experienced Nepali investors look at, added as switchable rules (`python -m nepse_kronos.investor_backtest --features ...`):

| Feature | Rule | Data (collector) | From |
|---|---|---|---|
| `money` | invest 50–100% of slots: +1 each if interbank rate < its 12-month average, credit/deposit ratio < its 12-month average, lending rate lower than 3 months earlier | NRB monthly tables, used 40 days after month-end (`python -m nepse_kronos.nrb`) | 2016-08 |
| `quality` | skip shares whose latest *published* quarterly report shows a loss, P/E > 40 or price > 5× book value | NEPSE company reports (`python -m nepse_kronos.nepse_reports`) | 2021-06 |
| `supply` | skip shares ≤ 60 days before promoter unlock or ≤ 30 days before a right-share book closure | ShareSansar (`python -m nepse_kronos.sharesansar`) | lock-ins ~2023 |
| `broker` | nudge ranking toward shares the top-5 net-buying brokers buy more than usual; skip possible pumping (top-5% buyer concentration + >20% rise in 20 days) | floorsheet (`python -m nepse_kronos.brokers`, ~3 GB) | 2024-01 |
| `sectors` | at most 5 of 20 holdings per sector | ShareSansar + MeroLagani | all |
| `taxwait` | hold a profitable share up to 60 days longer to reach the 7.5% (>1 year) tax rate | — | all |

Capital-gains tax is now 10% (≤ 1 year) / 7.5% (> 1 year).

**Missing companies fixed.** The original price data lacked merged/delisted companies (e.g. CTBNL, MEGA, PFC, CBL). `python -m nepse_kronos.history` builds `data/nepse/clean_full/`: listed companies keep the existing adjusted series; 316 merged/delisted companies are added from rajeevpaudel/nepse-history, adjusted with ShareSansar bonus/right/cash data, and 215 remaining one-day drops beyond the ±10% limit are repaired as unrecorded corporate actions. Rebuilding *all* series from raw prices was rejected: without merger swap ratios half the listed series drifted > 5% from the reference. Prices are total-return (cash dividends included), so the engine doesn't pay dividends separately (the 5% dividend tax is not modelled).

### Results (`data/nepse/clean_full`, rules fixed before testing)

| Window | Plain calm-shares system | + one feature | **All features** | NEPSE index |
|---|---|---|---|---|
| 2017-06 → 2026-10 | +49% (worst −37%) | money +66%, sectors +53%, supply +51%, taxwait +49% | **+81% (worst −23%), beat index 6/10 years** | +61% (worst −43%) |
| 2022-10 → 2026-10 | +13% | quality +15% | +21% | +36% |
| 2024-02 → 2026-10 | +11% | broker +12% | +16% | +25% |

- **The money cycle is the main source of value**: it cut exposure before the 2018 and 2022 falls.
- Every combination beat the plain system in every window, so paper trading now uses all features (`python -m nepse_kronos.paper_trade`; `--update-slow` monthly for company reports and ShareSansar lists).
- Nothing beat the index over 2022–2026, when the market rose strongly — the edge so far comes from avoiding big falls, which happen rarely; ~2 such episodes drive the 9-year result.

Caveats: the delisted series rely on our own adjustments plus gap repair; quality/broker data cover only recent years; NRB release dates before 2020 are approximated as month-end + 40 days; ShareSansar right-share history is refreshed only when re-collected.

### Update: fair yardstick, fixed recent prices, interest on cash, robustness (October 2026)

- **Missed adjustments fixed:** the source's adjusted prices hadn't applied some recent bonus/dividend events (e.g. NABIL 2025-12-31 and 2026-09-30 showed fake drops). `nepse_kronos.history` now compares them with official unadjusted closes and applied 196 missing adjustments.
- **Fair yardstick:** the NEPSE index has no dividends, our prices do. The benchmark is now *index + estimated dividends* (average cash yield of the 150 most-traded shares, ~0.6%/year; `data/nepse/meta/market_dividend_yield.csv`).
- **Interest on idle cash** (`interest` feature): banks' weighted-average deposit rate from NRB, minus 6% tax.

| 2017-06 → 2026-10 | Result | Worst drop |
|---|---|---|
| NEPSE index + dividends (fair yardstick) | +71% | −43% |
| Plain calm-shares system (+ interest) | +67% | −37% |
| All features, no interest | +94% | −23% |
| **All features + interest** | **+130%** | **−19%** |

**Robustness** (all features + interest, one setting changed at a time; `outputs/nepse/robust/summary.csv`): 15 of 16 variants beat the fair yardstick, but by +5 to +80 points (typical ≈ +30, i.e. ~1.5–2%/year) — the reference settings sit on the lucky side. Worst drop stayed −19% to −26% in every variant (index −43%). By start year: ahead from 2018, 2019 and 2021; **behind from 2020 (−15 points)**, because the system lags strong booms like 2020–21.

**Verdict:** the reliable benefit is *much smaller crashes* with roughly market-like or modestly better returns; the size of the extra return is uncertain and comes mostly from avoiding the 2018 and 2022 falls and earning interest while out of the market.

### Automatic daily run (macOS)

`scripts/daily_paper_trade.sh` runs the full update + paper portfolio and publishes that day's report to
`reports/paper/` (`LATEST.md`, one file per trading day, `trades.csv`) in the personal GitHub repo. It is scheduled
by `~/Library/LaunchAgents/com.sunilaryal.nepse-kronos.daily.plist` for Monday–Friday at 17:00 (Mac local time);
if the Mac is asleep at 17:00 it runs on wake. Logs: `outputs/nepse/daily.log`.

    launchctl kickstart gui/$(id -u)/com.sunilaryal.nepse-kronos.daily     # run now
    launchctl bootout gui/$(id -u)/com.sunilaryal.nepse-kronos.daily       # stop the schedule
