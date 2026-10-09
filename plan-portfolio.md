# NEPSE Trading System Backtest — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The final code lives in the files named in each task; this plan records the rules, interfaces, test cases and commands.

**Goal:** Find out whether a rules-based NEPSE portfolio — choosing *when to buy, what, how much, and when to sell* — beats simply holding the NEPSE index after real fees and tax, and whether adding Kronos to the stock ranking makes it better.

**Approach (plain English):** Every trading day, after prices come in, check each share we own against fixed selling rules. Sell the next morning if a rule says so. Fill empty slots with the best-ranked share we don't own. Refresh the full ranking every 10 trading days. Test simple ranking rules over 2017–2026 (fast), then Kronos over the last 2 years (slow), on the same rules.

**Decisions from the user (2026-10-04):** capital Rs 10 lakh; check daily and act only when a rule says so; loss limit = 15% below the best price since buying; run Kronos (~2 h); no git commits.

**Tech Stack:** Python 3.11 (`.venv311`), pandas, numpy, torch, matplotlib, pytest; existing `nepse_kronos` package.

## Global Constraints

- No git commits, merges or pushes. Each task ends with a checkpoint (tests green) instead of a commit.
- Activate `.venv311`, run everything from the repo root.
- No look-ahead: decisions use data up to a day's close; trades happen at the next market day's open.
- Long-only, whole shares, market calendar = `NEPSE_INDEX` dates.
- Price limit: no buy if a share opens ≥ 9.5% up; no sell if it opens ≥ 9.5% down (retry next day).
- Costs (resident individual, checked 2026-10-04): broker commission 0.36% (≤ Rs 50k), 0.33% (≤ 5 lakh), 0.31% (≤ 20 lakh), 0.27% (≤ 1 crore), 0.24% above; SEBON 0.015% each side; DP Rs 25 per stock per trade; capital-gains tax 10% on profit (holding ≤ 1 year); slippage 0.25% plus an extra cost that grows with trade size relative to the share's normal daily trading.
- Sale money can be reused only after 2 trading days (T+2 settlement).
- Unit tests never load Kronos weights.

## Trading rules (fixed before testing; do not tune on results)

| Rule | Value |
|---|---|
| Universe | 150 most-traded shares (median daily turnover over 60 days) with ≥ 128 days of history |
| Ranking refresh | every 10 trading days |
| Portfolio | up to 20 shares |
| Buy | fill empty slots with the best-ranked shares not owned, next morning |
| Keep zone | keep a share while it stays in the top 50 of the latest ranking |
| Loss limit | sell if the close is ≥ 15% below the highest close since buying |
| Time limit | sell if held ≥ 60 trading days and still at or below the buy price |
| Cool-off | don't re-buy a share within 20 trading days of selling it |
| Size | Rs 10 lakh / 20 per share, reduced for jumpy shares (× median volatility ÷ the share's volatility, capped at 1) and capped at 10% of the share's median daily turnover |

## Ranking signals compared

| Name | What it means | Period |
|---|---|---|
| `momentum` | biggest rise over the last 60 days | 2017-06 → 2026-10 |
| `steady` | rise over the last 60 days ÷ how jumpy the share was | 2017-06 → 2026-10 |
| `lowvol` | least jumpy shares | 2017-06 → 2026-10 |
| `kronos` | Kronos-small predicted 10-day return (lookback 128, 10 samples) | 2024-10 → 2026-10 |

All four are also scored on the common period 2024-10 → 2026-10 so Kronos is compared like for like.

## Decision rules (fixed in advance)

Kronos ranking is **useful** only if, on 2024-10 → 2026-10, its portfolio (1) beats the index and every simple signal after costs, (2) does so in both years separately, and (3) its rankings line up with what actually happened (average rank correlation > 0 with t-stat ≥ 2).
A simple signal is **promising** if, over 2017–2026, it beats the index after costs in most years with a smaller worst drop.

## File Structure

| Path | Responsibility |
|---|---|
| `nepse_kronos/universe.py` | panel loading, liquid universe, refresh dates |
| `nepse_kronos/signals.py` | `kronos`, `momentum`, `steady`, `lowvol` rankings; on-disk cache |
| `nepse_kronos/costs.py` | NEPSE fees, tax, size-dependent slippage |
| `nepse_kronos/portfolio.py` | daily rules engine (`Rules`, `simulate`) |
| `nepse_kronos/metrics.py` | returns, worst drop, Sharpe, turnover, rank correlation, per-year table |
| `nepse_kronos/portfolio_backtest.py` | CLI; saves `equity.csv`, `trades.csv`, `ic.csv`, `equity.png` |
| `tests/nepse_pipeline/test_*.py` | unit tests per module (+ `StubPredictor.predict_batch` in `conftest.py`) |
| `nepse_kronos/README.md` | "Trading system backtest" section with results |

---

### Task 1: Universe
**Files:** create `nepse_kronos/universe.py`, `tests/nepse_pipeline/test_universe.py`
**Interfaces:** `load_panel(clean_dir, include_indices=False) -> dict[str, DataFrame indexed by date]`; `liquid_universe(panel, as_of, top_n, min_history=150, liquidity_window=60) -> list[str]`; `rebalance_dates(calendar, start, end, every) -> list[Timestamp]`
**Tests:** panel indexed by date and skips `_INDEX` files; universe ranks by median turnover using only data ≤ as_of (a share liquid only later is not ranked up; shares that stopped trading or have short history are excluded; ties alphabetical); refresh dates every N market days.
- [x] Write tests → fail → implement → pass → full suite green.

### Task 2: Signals
**Files:** create `nepse_kronos/signals.py`, `tests/nepse_pipeline/test_signals.py`; modify `tests/nepse_pipeline/conftest.py` (add `StubPredictor.predict_batch`, forecasting close = last × last/first)
**Interfaces:** `kronos_signal(predictor, panel, symbols, as_of, future_dates, lookback=128, sample_count=10, batch_size=32)`; `momentum_signal(panel, symbols, as_of, window=60)`; `steady_signal(panel, symbols, as_of, window=60)`; `lowvol_signal(panel, symbols, as_of, window=60)`; `SIMPLE_SIGNALS = {"momentum", "steady", "lowvol"}`; `cached_signal(cache_dir, as_of, compute)`. All return a Series named `pred_return` (higher = better), index = symbol, sorted.
**Tests:** Kronos predicted returns match the stub, are batched (sizes `[2, 1]` for batch_size 2), ignore data after as_of, skip short histories; momentum/steady/lowvol values and ordering on synthetic up/flat/down/jumpy shares; cache computes once.
- [x] Write tests → fail → implement → pass → full suite green.

### Task 3: Costs
**Files:** create `nepse_kronos/costs.py`, `tests/nepse_pipeline/test_costs.py`
**Interfaces:** `CostModel(commission_tiers, sebon_rate=0.00015, dp_charge=25.0, slippage=0.0025, impact=0.1, cgt_rate=0.10)` with `commission(value)`, `trade_cost(value, daily_turnover=None)` (slippage rate = `slippage + impact × value / daily_turnover`, capped at 5%), `capital_gains_tax(proceeds, cost_basis)`
**Tests:** commission tiers at boundaries; trade cost formula with and without daily turnover; tax only on profit.
- [x] Write tests → fail → implement → pass → full suite green.

### Task 4: Daily rules engine
**Files:** create `nepse_kronos/portfolio.py`, `tests/nepse_pipeline/test_portfolio.py`
**Interfaces:** `Rules` dataclass (`top_k=20, keep_rank=50, stop_loss=0.15, time_stop_days=60, cooldown_days=20, max_turnover_share=0.10, vol_window=60, settlement_days=2, size_by_volatility=True`); `simulate(panel, calendar, signals, rules, capital=1_000_000, costs=None) -> (equity Series, trades DataFrame)`. `trades` columns: `date, symbol, side, shares, price, value, fees, tax, reason` (`reason` ∈ `rank`, `stop_loss`, `time_limit`, `buy`).
**Daily loop:** (1) at the open, release settled cash, execute queued sells then queued buys (price-limit checks); (2) at the close, value the portfolio, update each holding's highest close; (3) queue sells for: loss limit, time limit, or (on refresh days) rank worse than `keep_rank` / missing from the ranking; (4) queue buys for empty slots from the latest ranking, skipping owned, queued and cooling-off shares.
**Tests:** buys the top pick at the next open; keep zone (a share slipping from #1 to #3 with keep_rank 3 is kept, to #4 is sold); 15% loss limit from the highest close triggers a sale next morning; time limit; cool-off blocks re-buy; T+2 (sale money not usable for 2 days); upper-limit open blocks a buy; sale retried when a share doesn't trade; size cut for jumpy shares and capped by turnover; fees and tax match `CostModel`.
- [x] Write tests → fail → implement → pass → full suite green.

### Task 5: Metrics
**Files:** create `nepse_kronos/metrics.py`, `tests/nepse_pipeline/test_metrics.py`
**Interfaces:** `TRADING_DAYS_PER_YEAR = 240`; `performance(equity, benchmark)`; `trade_summary(trades, equity)` (+ counts by reason); `rank_ic(signals, panel, calendar, horizon)`; `summarize_ic(ic)`; `yearly_table(equity, benchmark) -> DataFrame` (per calendar year: strategy %, index %).
**Tests:** total return, worst drop, CAGR over one year, flat benchmark Sharpe is NaN; perfect/reversed rank correlation; trade summary sums; yearly table values.
- [x] Write tests → fail → implement → pass → full suite green.

### Task 6: Backtest command
**Files:** create `nepse_kronos/portfolio_backtest.py`, `tests/nepse_pipeline/test_portfolio_backtest.py`
**Interfaces:** `python -m nepse_kronos.portfolio_backtest --signal {kronos,momentum,steady,lowvol} [--start] [--end] [--refresh-every 10] [--horizon 10] [--universe 150] [--top-k 20] [--keep-rank 50] [--stop-loss 0.15] [--time-stop 60] [--capital 1000000] [--sample-count 10] [--model] [--tokenizer] [--tag]`; writes `<out-dir>/<tag>/{equity.csv, trades.csv, ic.csv, equity.png}` and prints the performance table, per-year table, trade summary and ranking quality. Signal cache in `<out-dir>/signals/<signal-tag>/` (keyed by signal settings only, so rule changes reuse forecasts).
**Tests:** end-to-end on synthetic data with `--signal momentum` (no model): output files exist with the right columns; "sharpe" printed; one cache file per refresh date.
- [x] Write test → fail → implement → pass → full suite green.

### Task 7: Run the experiments
- [x] Refresh data: `python -m nepse_kronos.fetch && python -m nepse_kronos.prepare > /dev/null`
- [x] Simple signals, 2017-06-01 → latest: `python -m nepse_kronos.portfolio_backtest --signal momentum --start 2017-06-01` (and `steady`, `lowvol`)
- [x] Ranking stability check for Kronos at 10 samples (two runs on one date; agreement ≥ 0.8, top-20 overlap ≥ 14/20; else use 20 samples)
- [x] Kronos, 2024-10-01 → latest, in the background: `python -m nepse_kronos.portfolio_backtest --signal kronos --start 2024-10-01` (~2 h, resumable)
- [x] Simple signals on the same 2024-10-01 → latest window (for like-for-like comparison)
- [x] Apply the decision rules above; write results + verdict + caveats into `nepse_kronos/README.md`

## Outcome (2026-10-04)

All tasks done; 57 tests pass. Ranking stability at 10 samples was 0.80 agreement / 12 of 20 overlap, so Kronos ran with 20 samples (46 refreshes, ~1 h 40 min). Results and verdict are in `nepse_kronos/README.md` section 7: no signal beat holding the index under the agreed rules; Kronos was the worst (−34% vs −1.4%) because its top 20 reshuffles ~75% every two weeks; the calmest-shares ranking is the only reliable signal found, and the 15% loss limit was the most harmful rule.

## Later phases (need new data; separate plans)

- **D. Industry limits & rotation** — sector list for each share (scrape).
- **E. Broker activity** — floorsheet data (already in the source dataset, not downloaded).
- **F. Company news and health** — bonus/dividend/book-closure dates, quarterly profits, P/E, bank bad loans.
- **G. When to be in the market** — interest rates, bank lending capacity, NRB margin rules. (A simple "index above its 100/200-day average" switch was tested on 2017–2026 and did worse than holding: +39–46% vs +71%.)
