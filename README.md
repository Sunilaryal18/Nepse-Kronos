# Nepse-Kronos

A research toolkit for the **Nepal Stock Exchange (NEPSE)**: free public data, honest backtests, a rules-based
investing system, daily paper trading and a one-page dashboard.

> Research and analysis only — **not investment advice**. Past results don't guarantee future ones.

## What it does

- **Collects NEPSE data** from public sources: daily prices (including merged and delisted companies), the NEPSE
  index, dividends and bonus/right shares, sectors, quarterly company reports, NRB interest-rate and bank-lending
  data, and broker-level trades.
- **Tests investing ideas honestly** — with real NEPSE costs (broker commission, SEBON fee, DP charge, capital-gains
  tax), next-day trading, the ±10% daily price limit, T+2 settlement, no look-ahead, and a benchmark that includes
  dividends.
- **Runs a paper portfolio every weekday at 5 pm** and publishes the day's report.
- **Builds a dashboard** with the paper portfolio vs the market, tomorrow's orders, today's top movers, the top 5
  shares to consider at today's prices (each with a candlestick chart and plain-English *Why* and *Risks*), and —
  on your own computer only — your real holdings with a health check per share.

## Quick start

Requires macOS or Linux and Python 3.11.

```bash
uv venv --python 3.11 .venv311 && source .venv311/bin/activate
uv pip install -r requirements.txt -r requirements-nepse.txt

python -m nepse_kronos.fetch                 # prices and indices (public datasets)
python -m nepse_kronos.prepare               # clean them
python -m nepse_kronos.sharesansar           # dividends, right shares, sectors, IPOs (slow: ~1 page/second)
python -m nepse_kronos.nepse_reports         # quarterly company reports (NEPSE website)
python -m nepse_kronos.nrb                   # interest rates, credit-to-deposit ratio, remittances
python -m nepse_kronos.brokers               # broker-level trade data (~3 GB, 2024 onward)
python -m nepse_kronos.history               # complete, adjusted price set incl. delisted companies

python -m nepse_kronos.paper_trade           # daily paper portfolio (updates data first)
python -m nepse_kronos.dashboard --open      # the dashboard
```

## Daily use

| Task | Command |
|---|---|
| Update everything and run the paper portfolio | `python -m nepse_kronos.paper_trade` (add `--update-slow` about once a month) |
| Open the dashboard | `python -m nepse_kronos.dashboard --open` |
| Check your real holdings | put them in `data/nepse/my_holdings.csv`, then `python -m nepse_kronos.my_portfolio` |
| Run it automatically at 5 pm on weekdays | `scripts/daily_paper_trade.sh` with a launchd job — see [the guide](nepse_kronos/README.md#automatic-daily-run-macos) |
| Test a strategy | `python -m nepse_kronos.investor_backtest --features money,sectors,supply,taxwait,interest` |

## How the system works

**Paper portfolio** — hold up to 20 of the calmest among the 150 most-traded shares (ranking refreshed every 10
trading days), keep a share while it stays in the top 50, no automatic stop-loss, plus:

| Rule | What it does |
|---|---|
| Money cycle | Invest 50–100% depending on NRB interest rates and bank lending room (published data only) |
| Quality | Skip shares with a published loss, P/E above 40 or price above 5× book value |
| New supply | Skip shares shortly before promoter unlocks or right-share book closures |
| Brokers | Prefer shares the biggest brokers are buying; skip likely pumping |
| Sectors | At most 5 shares from one sector |
| Tax wait / interest | Wait for the lower 1-year tax rate when close; idle cash earns the bank deposit rate |

**Daily picks** — profit per share up >10% on a year ago, P/E ≤ 25, price ≤ 2.5× book value, not in a falling trend,
not overheated, big brokers not selling; calmest first, with a buying range from support and the 20-day average.

## What the research found

Every rule was written down before testing. Highlights (details in [the guide](nepse_kronos/README.md)):

| Test | Result |
|---|---|
| Kronos price forecasts on NEPSE | Did **not** beat "no change" |
| Kronos fine-tuned on NEPSE (held-out year) | No clear improvement; picked up a downward bias |
| Kronos as a stock picker (2024–2026) | **−34%** vs −1.4% for the index — lost to every simple ranking rule, mainly by reshuffling its picks and over-trading |
| Simple trading rules with a 15% stop-loss | Lost to the index — NEPSE shares often fall 15% and recover; fees and tax took about half of gains |
| Calm-shares system + all rules + interest, 2017–2026 | **+130% vs +71%** for the index with dividends; worst fall −19% vs −43% |
| Same system with 16 small rule changes | 15 of 16 beat the index, typically by ~1.5–2% a year; **smaller crashes every time**; behind the index in strong booms |
| Daily picks screen, each pick held one year (2022–2025) | Beat the market 45–50% of the time, about +1% a year on average — sound candidates, not predictions |

The dependable benefit is **much smaller losses in crashes**, mainly from the money-cycle rule.

## Data quality

Backtests are only as honest as their data. Problems found and fixed:

- **Survivorship bias:** the main price dataset only covered companies still listed today. **316 merged or delisted
  securities** were added back from a second public dataset, so tests no longer see only the survivors.
- **Missed corporate-action adjustments:** the source's adjusted prices hadn't applied some bonus and dividend
  events, leaving fake price drops (e.g. NABIL, 30 Sep 2026). Comparing with official unadjusted prices found and
  applied **~200 missed adjustments** (198 in the latest build), checked again on every update.
- **Unrecorded corporate actions:** in the added delisted series, **215 one-day drops beyond NEPSE's ±10% daily
  limit** — impossible as real trading — were repaired as missing bonus/right adjustments.
- **Close-only history:** index data before late 2016 has no real candles and is trimmed; mutual funds (Rs 10 face
  value) are kept out of dividend estimates.
- **No look-ahead:** company reports count from their publication date and NRB data from about 40 days after
  month-end; trades happen at the next day's open.

## Project layout

```
nepse_kronos/        NEPSE toolkit: data collectors, price build, strategy rules, trading engine,
                     backtests, paper trading, picks, dashboard, holdings tracker (detailed guide: README.md inside)
tests/nepse_pipeline/  Tests for everything above (no network needed)
scripts/             Daily 5 pm job
reports/paper/       Published daily paper-portfolio reports and dashboard
model/               Kronos model code (from the original project)
finetune/, finetune_csv/, examples/, webui/   Original Kronos training, examples and web UI
docs/KRONOS.md       The original Kronos README
```

Local data (`data/`), model weights and outputs are not stored in git.

## Credits

- **[Kronos](https://github.com/shiyu-coder/Kronos)** by Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu,
  Changshui Zhang and Jian Li — the foundation model for financial candlesticks this project started from (MIT licence).
  Model weights: [NeoQuasar on Hugging Face](https://huggingface.co/NeoQuasar). Original README: [docs/KRONOS.md](docs/KRONOS.md).
  If you use the model, please cite their paper:

  ```
  @misc{shi2025kronos,
        title={Kronos: A Foundation Model for the Language of Financial Markets},
        author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
        year={2025}, eprint={2508.02739}, archivePrefix={arXiv}, primaryClass={q-fin.ST},
        url={https://arxiv.org/abs/2508.02739},
  }
  ```
- **Data:** [socrateai-official/nepse-open-data](https://github.com/socrateai-official/nepse-open-data) (MIT) and
  [rajeevpaudel/nepse-history](https://github.com/rajeevpaudel/nepse-history) (MIT) for prices and trades;
  [Nepal Rastra Bank](https://www.nrb.org.np) for macroeconomic data; the [NEPSE](https://www.nepalstock.com) website
  via [NepseUnofficialApi](https://github.com/basic-bgnr/NepseUnofficialApi); [ShareSansar](https://www.sharesansar.com)
  and [MeroLagani](https://merolagani.com) for dividends, sectors and issues (collected politely for personal
  research; the collected data is not redistributed here).

## License

MIT — see [LICENSE](LICENSE). The original Kronos copyright notice is retained.
