# Nepse-Kronos

Tools for researching the Nepal Stock Exchange (NEPSE). I started from the open-source Kronos model to see whether
a financial foundation model could pick NEPSE shares. It couldn't, so the project grew into a data pipeline,
a backtester that charges real NEPSE fees and taxes, and a paper portfolio that updates every trading day.

Nothing here is investment advice. It's research, and past results won't necessarily repeat.

## What's in it

The data side downloads daily prices for every listed share and index, including companies that have since merged
or delisted. It also collects dividends and bonus/right shares, sectors, quarterly company reports, Nepal Rastra
Bank interest-rate and lending data, and broker-level trades.

The backtester charges broker commission, the SEBON fee, the DP charge and capital-gains tax. Trades fill at the
next day's open, the ±10% daily price limit and T+2 settlement apply, and no data is used before it was published.
The benchmark is the NEPSE index plus an estimate of its dividends, because the share prices already include them.

A paper portfolio runs every weekday at 5 pm and publishes its report here, in `reports/paper/`.

The dashboard shows the paper portfolio against the index, tomorrow's orders, the day's biggest movers, and five
shares worth a look at today's prices. Each of those comes with a candlestick chart and a short list of reasons and
risks. If you add your own holdings, the local copy of the dashboard shows them too, with a health check for each
share. That part never gets published.

## Getting started

You need macOS or Linux and Python 3.11.

```bash
uv venv --python 3.11 .venv311 && source .venv311/bin/activate
uv pip install -r requirements.txt -r requirements-nepse.txt

python -m nepse_kronos.fetch                 # prices and indices from public datasets
python -m nepse_kronos.prepare               # clean them
python -m nepse_kronos.sharesansar           # dividends, right shares, sectors, IPOs (slow, about 1 page a second)
python -m nepse_kronos.nepse_reports         # quarterly company reports from the NEPSE website
python -m nepse_kronos.nrb                   # interest rates, credit-to-deposit ratio, remittances
python -m nepse_kronos.brokers               # broker-level trades since 2024 (about 3 GB)
python -m nepse_kronos.history               # one adjusted price set, delisted companies included

python -m nepse_kronos.paper_trade           # paper portfolio (updates the data first)
python -m nepse_kronos.dashboard --open      # dashboard
```

## Day to day

| What | Command |
|---|---|
| Update the data and the paper portfolio | `python -m nepse_kronos.paper_trade` (add `--update-slow` once a month) |
| Open the dashboard | `python -m nepse_kronos.dashboard --open` |
| Check your own shares | list them in `data/nepse/my_holdings.csv`, then run `python -m nepse_kronos.my_portfolio` |
| Run it every weekday at 5 pm | `scripts/daily_paper_trade.sh` plus a launchd job, see [the guide](nepse_kronos/README.md#automatic-daily-run-macos) |
| Backtest a strategy | `python -m nepse_kronos.investor_backtest --features money,sectors,supply,taxwait,interest` |

## How the paper portfolio picks shares

It holds up to 20 shares, chosen as the calmest of the 150 most-traded, and re-ranks every 10 trading days.
A share stays until it drops out of the top 50. There's no automatic stop-loss. On top of that:

| Rule | What it does |
|---|---|
| Money cycle | Invests 50 to 100% of the money depending on NRB interest rates and how much room banks have to lend |
| Quality | Skips shares with a reported loss, a P/E above 40, or a price above 5 times book value |
| New supply | Skips shares just before promoter shares unlock or a right-share book closure |
| Brokers | Prefers shares the biggest brokers are buying and skips ones that look pumped |
| Sectors | Holds at most 5 shares from any one sector |
| Tax and cash | Holds a winner a little longer if that gets the lower one-year tax rate, and earns deposit interest on idle cash |

The five daily picks use a separate screen. Profit per share has to be up more than 10% on a year earlier, with a
P/E of 25 or less and a price no more than 2.5 times book value. The share can't be in a falling trend or overheated,
and the big brokers can't be selling it. Picks are sorted calmest first, and the buying range comes from recent
support and the 20-day average.

## Results

I wrote each rule down before testing it, so the numbers weren't tuned after the fact. The
[guide](nepse_kronos/README.md) has the details.

| Test | Result |
|---|---|
| Kronos price forecasts | No better than assuming tomorrow's price equals today's |
| Kronos fine-tuned on NEPSE, tested on a held-out year | No clear improvement, plus a downward bias |
| Kronos as a stock picker, 2024 to 2026 | Lost 34% while the index lost 1.4%. It reshuffled its picks so often that fees ate it alive |
| Simple trading rules with a 15% stop-loss | Behind the index. NEPSE shares often fall 15% and then recover, and fees and tax took about half the gains |
| Calm shares with all the rules and deposit interest, 2017 to 2026 | Up 130% against 71% for the index with dividends. Worst fall 19% against 43% |
| The same system with 16 small rule changes | 15 of the 16 still beat the index, usually by 1.5 to 2% a year. All 16 fell less in crashes. All of them lag in strong booms |
| The daily-picks screen, each pick held a year (2022 to 2025) | Beat the market 45 to 50% of the time, about 1% a year better on average. Treat picks as candidates, not predictions |

So the system mostly earns its keep in bad markets: it loses much less in crashes, and that comes mainly from the
money-cycle rule.

## Data problems I found and fixed

The main price dataset only covers companies that are still listed, which flatters any backtest. I added 316 merged
or delisted securities back from a second public dataset.

Its adjusted prices had also missed some recent bonus and dividend adjustments, which showed up as fake price drops
(NABIL on 30 September 2026, for example). Checking against the official unadjusted prices turned up about 200 of
these (198 in the latest build). The check runs again on every update.

In the delisted companies I added myself, 215 one-day drops were bigger than NEPSE's 10% daily limit allows. Real
trading can't do that, so I treated them as bonus or right-share adjustments nobody recorded and corrected them.

A few smaller things: index data before late 2016 has only closing prices, so it's trimmed. Mutual funds have a
Rs 10 face value, so they're left out of the dividend estimate. Company reports only count from the day they were
published, and NRB figures from about 40 days after the month they describe.

## Layout

```
nepse_kronos/          the NEPSE code: data collectors, price build, strategy rules, trading engine,
                       backtests, paper trading, picks, dashboard and holdings tracker (see its README)
tests/nepse_pipeline/  tests for all of it, no network needed
scripts/               the 5 pm job
reports/paper/         published daily reports and dashboard
model/                 the Kronos model (from the original project)
finetune/, finetune_csv/, examples/, webui/   Kronos training code, examples and web UI
docs/KRONOS.md         the original Kronos README
```

Downloaded data, model weights and other outputs stay out of git.

## Credits

[Kronos](https://github.com/shiyu-coder/Kronos) was built by Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu,
Changshui Zhang and Jian Li, and released under the MIT licence. This project started from their code. The model
weights are on [Hugging Face](https://huggingface.co/NeoQuasar), and their original README is in
[docs/KRONOS.md](docs/KRONOS.md). If you use the model, please cite their paper:

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025}, eprint={2508.02739}, archivePrefix={arXiv}, primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739},
}
```

Price and trade data come from [socrateai-official/nepse-open-data](https://github.com/socrateai-official/nepse-open-data)
and [rajeevpaudel/nepse-history](https://github.com/rajeevpaudel/nepse-history), both MIT-licensed. Macroeconomic data
comes from [Nepal Rastra Bank](https://www.nrb.org.np). Company reports and recent prices come from the
[NEPSE website](https://www.nepalstock.com), read with [NepseUnofficialApi](https://github.com/basic-bgnr/NepseUnofficialApi).
Dividends, sectors and share issues come from [ShareSansar](https://www.sharesansar.com) and
[MeroLagani](https://merolagani.com). I collected those slowly for personal research and don't republish the data here.

## License

MIT, see [LICENSE](LICENSE). The original Kronos copyright notice is kept.
