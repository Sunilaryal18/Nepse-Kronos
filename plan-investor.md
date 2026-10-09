# NEPSE "Top Investor" Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Data collectors (Tasks A–E) are independent and may run in parallel; integration (Task F onward) waits for them. TDD: every parser is tested on saved sample responses (no network in unit tests).

**Goal:** Add what experienced Nepali investors look at — dividends/bonus, company quality, the money cycle, sectors, new-share supply, broker behaviour — fix the missing-companies problem in our price data, and re-test the trading system honestly.

**Decisions from the user (2026-10-04):** collect ShareSansar/MeroLagani data politely (≤ 1 request/second, cached, personal research); use the unofficial NEPSE API package for company reports; broker data 2024 onward (~3 GB); report only at the end; no git commits.

**Tech:** Python 3.11 (`.venv311`), pandas, requests, beautifulsoup4, openpyxl, `nepse` (NepseUnofficialApi), pytest. New third-party packages go in `requirements-nepse.txt`.

## Global Constraints
- No git commits. All downloaded/derived data under `data/nepse/` (git-ignored). Raw web responses cached under `data/nepse/source/<site>/` so each page is fetched once.
- Web politeness: ≥ 1 s between requests per site, a descriptive User-Agent, retries with back-off, resume from cache.
- **No look-ahead:** every fundamental/macro/event value carries the date it became public (`available_from`). Backtests may only use rows with `available_from` ≤ decision date.
- Unit tests never touch the network; they parse fixture files saved under `tests/nepse_pipeline/fixtures/`.
- Don't modify `tests/nepse_pipeline/conftest.py` or existing modules from a data-collector task; integration tasks own those changes.

## Data sources (verified 2026-10-04)
| Data | Source | Coverage |
|---|---|---|
| Dividends (cash %, bonus %), book-closure date | ShareSansar `/company-dividend` (POST, CSRF token, company id from page) | 2011→ |
| Right shares (ratio, book closure) | ShareSansar `/company-rightshare` | 2011→ |
| Sectors (listed) | ShareSansar `/company-list?sector=ID` (JSON, 50/page) | current |
| Sectors (delisted/merged) | MeroLagani `AutoSuggestHandler.ashx?type=Company` + `CompanyDetail.aspx?symbol=` | all 1,676 symbols |
| IPO / right / FPO calendar; promoter lock-ins | ShareSansar `/existing-issues?type=T`, `/promoter-lockin` | 2011→ / ~2023→ |
| Quarterly EPS, P/E, net worth/share, submit date | NEPSE `/api/nots/application/reports/<id>` via `nepse` package | FY 2078/79 (2022)→ |
| Interest rates, remittances | NRB "Current Macroeconomic and Financial Situation – Tables" xlsx (A22, A7) | Aug 2016→ |
| Credit-to-deposit ratio | NRB "Banking and Financial Statistics" xlsx (C18) | Aug 2016→ |
| Prices incl. delisted companies (unadjusted) | rajeevpaudel/nepse-history `data/ohlcv/` (MIT) | 2015→ |
| Broker-level trades | socrateai-official/nepse-open-data `floorsheet/` (MIT) | 2024-01→ |

## Tasks

### A. ShareSansar + MeroLagani collector — `nepse_kronos/sharesansar.py`, `nepse_kronos/merolagani.py`
Outputs (`data/nepse/meta/`): `corporate_actions.csv` (symbol, fiscal_year, cash_pct, bonus_pct, book_close, announced), `rights.csv` (symbol, ratio_new_per_old, book_close), `sectors.csv` (symbol, name, sector, listed: bool), `issues.csv` (symbol, issue_type, open_date, close_date, listing_date, units, price, ratio), `lockins.csv` (symbol, allot_date, promoter_lock_end, mf_lock_end). CLI: `python -m nepse_kronos.sharesansar`.

### B. NRB macro collector — `nepse_kronos/nrb.py`
Output `data/nepse/meta/macro_monthly.csv`: month (mid-month reference date), interbank, tbill91, deposit_rate, lending_rate, base_rate, cd_ratio, remittance (monthly, de-cumulated), available_from (= reference date + 40 days unless a real publish date is known). CLI: `python -m nepse_kronos.nrb`.

### C. NEPSE company reports — `nepse_kronos/nepse_reports.py`
Output `data/nepse/meta/fundamentals.csv`: symbol, fiscal_year, quarter, eps (annualised), pe, net_worth_per_share, submitted (date) — `available_from = submitted`. CLI: `python -m nepse_kronos.nepse_reports`.

### D. Survivorship-free adjusted prices — `nepse_kronos/history.py` (needs A)
Build `data/nepse/clean_full/<SYMBOL>.csv` (canonical columns) from rajeevpaudel OHLCV for every symbol incl. delisted, back-adjusted for bonus and right shares from A; plus `data/nepse/meta/cash_dividends.csv` (symbol, ex_date, cash_per_unit in adjusted units). Validate against the existing adjusted data (`data/nepse/clean/`) where both exist: report the share of symbols whose daily returns match within 0.5% on ≥ 98% of days, and list mismatching bonus dates.

### E. Broker features — `nepse_kronos/brokers.py`
Download floorsheet 2024→ (sparse checkout). Per (date, symbol), rolling 20 trading days: `top5_net_buy` (net units bought by the 5 biggest net-buying brokers ÷ total units traded), `buyer_hhi` and `seller_hhi` (concentration), `trades`. Output `data/nepse/meta/broker_features.csv.gz`.

### F. Engine integration (after A–E)
1. **Cash dividends:** on each ex-date, holders receive `cash_per_unit × units`, minus 5% dividend tax, credited 30 days later (approximate payment lag).
2. **Sector limit:** at most 5 of 20 holdings per sector (`Rules.max_per_sector = 5`).
3. **Company quality filter** (only where data exists, 2022→): skip shares whose latest *published* annualised EPS ≤ 0, or P/E > 40, or price/net-worth-per-share > 5. Shares with no published report yet are allowed (filter off) before 2022.
4. **New-supply filter:** don't buy a share within 60 days before its promoter lock-in ends or within 30 days before a right-share book closure.
5. **Money-cycle exposure:** each month compute a "money is easy" score from published NRB data: +1 if interbank rate < its 12-month average, +1 if CD ratio < its 12-month average, +1 if lending rate is falling vs 3 months ago. Invest `top_k` slots × (0.5 + score/6) → 50%–100% of the portfolio; the rest stays in cash.
6. **Broker signal (2024→):** rank = lowvol rank, with a tie-breaker bonus for positive `top5_net_buy`; exclude shares whose `buyer_hhi` is in the top 5% of the universe on the decision day while the price rose > 20% in 20 days (possible pumping).
All features are switchable in `Rules`; defaults reproduce the previous behaviour.

### G. Experiments (rules fixed above; no tuning on results)
1. Re-run the earlier best system (lowvol, keep 50, no stop-loss) on survivorship-free data, 2017-06 → latest, with cash dividends. Compare with the old result.
2. Add features one at a time on their own data windows (sector limit & supply filter & money cycle: 2017→; quality: 2022→; broker: 2024-02→), then all together.
3. Report per period: return vs index, worst drop, fees+tax, dividends received, years beating the index. Verdict + caveats in `nepse_kronos/README.md` §9. Update `paper_trade` to the final rule set only if it beats the plain version in its test window.

## Outcome (2026-10-04)

All tasks done; 168 tests pass; nothing committed. Deviations from the plan, with reasons:
- **D:** rebuilding every series from raw prices failed validation (no merger swap ratios: half the listed series drifted > 5%), so listed companies keep the reference adjusted series and only merged/delisted companies (316) are rebuilt, with gap repair for one-day drops beyond the ±10% limit.
- **F1 (cash dividends):** both price sources are total-return adjusted (checked on NICA 2023-10-03: (761.5 − 1.52)/1.29 = 589.13), so paying dividends in the engine would double count; the feature exists in the engine but is not used.
- **F6 (broker):** `top5_net_buy` is never negative, so the nudge compares each share with that day's median instead of zero.
- Added `--max-age-days` to `nepse_reports` so paper trading can pick up new quarterly reports.
Results and verdict: `nepse_kronos/README.md` §9. All features combined: +81% vs index +61% (2017–2026), worst drop −23% vs −43%; behind the index in 2022–2026 and 2024–2026. Paper trading switched to all features.

## Follow-up (2026-10-04): data fix, fair benchmark, interest, robustness
See README section 9 "Update". 196 missed adjustments applied; benchmark = index + estimated dividends; `interest` feature added (in paper-trading defaults); 16 one-at-a-time variants: 15/16 beat the fair benchmark (typical +30 points over 9 years), worst drop -19% to -26% vs -43%; behind when starting in 2020.
