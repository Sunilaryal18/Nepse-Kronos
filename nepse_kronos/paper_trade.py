"""Daily paper trading: what do we own, how is it doing, and what to buy or sell tomorrow morning.

Run once per trading day after the day's prices are published. The first run starts a paper
portfolio on the latest trading day and saves its settings in <out-dir>/config.json; later runs
replay the same portfolio from that start date with the new data.

Rules: rank the 150 most-traded shares from calmest to jumpiest (refreshed every 10 trading days),
hold up to 20, keep a share while it stays in the top 50, no automatic stop-loss, real NEPSE costs,
plus the investor features chosen on the first run (default: all those that beat the plain system in
the 2017-2026 backtests - money cycle, sector limit, new-supply filter, tax wait, quality, broker).

Usage:
    python -m nepse_kronos.paper_trade                  # daily: new prices, broker data, NRB data; then report
    python -m nepse_kronos.paper_trade --update-slow    # ~monthly: also company reports and ShareSansar lists
    python -m nepse_kronos.paper_trade --no-update      # use the data already on disk
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import pandas as pd

from nepse_kronos.costs import CostModel
from nepse_kronos.investor_backtest import (FEATURES, NO_LIMIT, apply_features, load_sectors, lowvol_signals,
                                            parse_features, tradable_stocks)
from nepse_kronos.portfolio import Rules, simulate
from nepse_kronos.universe import load_panel, rebalance_dates

SELL_REASONS = {"rank": "no longer among the calmest 50", "stop_loss": "fell too far", "time_limit": "held too long"}
DEFAULT_FEATURES = "money,sectors,supply,taxwait,quality,broker,interest"


def update_data(clean_dir, slow=False):
    """Refresh everything the rules use. A failing source is reported, not fatal: older data is used."""
    from nepse_kronos import brokers, fetch, history, live_prices, nepse_reports, nrb, prepare, sharesansar
    steps = [("community prices", lambda: (fetch.main([]), prepare.main([]))),
             ("NEPSE website prices", lambda: live_prices.main([])),   # fills days the community data lacks
             ("price build", lambda: history.main(["--out-dir", str(clean_dir)])),
             ("broker data", lambda: brokers.main([])),
             ("NRB money data", lambda: nrb.main([]))]
    if slow:
        steps += [("company reports", lambda: nepse_reports.main(["--refresh-list", "--max-age-days", "7"])),
                  ("ShareSansar lists", lambda: sharesansar.main(["--refresh-lists"]))]
    for name, step in steps:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                step()
        except Exception as exc:  # keep going with the data already on disk
            print(f"Warning: could not update {name}: {exc}")


def load_config(path, args, latest):
    if path.exists():
        return json.loads(path.read_text())
    config = {
        "start": f"{latest:%Y-%m-%d}", "capital": args.capital, "signal": "lowvol", "window": 60,
        "universe": args.universe, "min_history": args.min_history, "top_k": args.top_k,
        "keep_rank": args.keep_rank, "refresh_every": 10, "stop_loss": None, "time_limit": None,
        "features": [f for f in args.features.split(",") if f],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return config


def report(config, stocks, index_close, equity, trades, state, latest, next_refresh_in, notes=()):
    def last_price(symbol):
        return stocks[symbol].loc[:latest, "close"].iloc[-1]

    start, capital = pd.Timestamp(config["start"]), config["capital"]
    value = equity.iloc[-1]
    index_change = index_close.loc[latest] / index_close.loc[start] - 1
    lines = [f"# Paper portfolio — {latest:%Y-%m-%d}", "",
             f"Started {start:%Y-%m-%d} with Rs {capital:,.0f}.",
             f"Value now: **Rs {value:,.0f}** ({value / capital - 1:+.1%}). "
             f"NEPSE index over the same days: {index_change:+.1%}.", "", *notes, *([""] if notes else [])]

    holdings = state["holdings"]
    if holdings:
        buys = trades[trades["side"] == "buy"].groupby("symbol").last()
        lines += [f"You own {len(holdings)} shares:", "",
                  "| Share | Units | Bought at | Last price | Value (Rs) | Change |", "|---|---|---|---|---|---|"]
        for symbol in sorted(holdings):
            price = last_price(symbol)
            bought = buys.loc[symbol, "price"]
            lines.append(f"| {symbol} | {holdings[symbol]} | {bought:,.1f} | {price:,.1f} | "
                         f"{holdings[symbol] * price:,.0f} | {price / bought - 1:+.1%} |")
    else:
        lines.append("You own no shares yet.")
    lines += ["", f"Cash: Rs {state['cash']:,.0f}" +
              (f" (+ Rs {state['unsettled']:,.0f} from recent sales, usable in 2 trading days)" if state["unsettled"] else ""),
              "", "## Orders for the next trading morning", ""]

    orders = [f"- SELL all {holdings[symbol]} units of {symbol} — {SELL_REASONS.get(reason, reason)}"
              for symbol, reason in sorted(state["sell_orders"].items())]
    for symbol, budget in state["buy_orders"]:
        price = last_price(symbol)
        orders.append(f"- BUY {symbol} for about Rs {budget:,.0f} (~{int(budget // price)} units at around {price:,.1f})")
    lines += orders or ["- Nothing to do."]
    lines += ["", f"Next ranking refresh in {next_refresh_in} trading day(s)."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", default="data/nepse/clean_full",
                        help="prices with missed bonus/dividend adjustments fixed (built by nepse_kronos.history)")
    parser.add_argument("--out-dir", default="outputs/nepse/paper")
    parser.add_argument("--meta-dir", default="data/nepse/meta")
    parser.add_argument("--no-update", action="store_true", help="Don't download new data first")
    parser.add_argument("--update-slow", action="store_true", help="Also refresh company reports and ShareSansar lists")
    parser.add_argument("--benchmark", default="NEPSE_INDEX")
    # used only on the first run, then frozen in config.json
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--universe", type=int, default=150)
    parser.add_argument("--min-history", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--keep-rank", type=int, default=50)
    parser.add_argument("--features", default=DEFAULT_FEATURES, help=",".join(FEATURES))
    args = parser.parse_args(argv)
    parse_features(parser, args.features)

    if not args.no_update:
        update_data(args.clean_dir, slow=args.update_slow)
    out = Path(args.out_dir)
    panel = load_panel(args.clean_dir, include_indices=True)
    index_close = panel[args.benchmark]["close"]
    sectors = load_sectors(args.meta_dir)
    stocks = tradable_stocks(panel, sectors)
    calendar = index_close.index
    latest = calendar[-1]
    config = load_config(out / "config.json", args, latest)

    dates = rebalance_dates(calendar, config["start"], latest, config["refresh_every"])
    signals = lowvol_signals(stocks, dates, out / "signals", config["universe"], config["min_history"],
                             config["window"])

    features = config.get("features", [])
    signals, extra, overrides = apply_features(features, stocks, signals, args.meta_dir, sectors)
    rules = Rules(top_k=config["top_k"], keep_rank=config["keep_rank"],
                  stop_loss=config["stop_loss"] or 1.0, time_stop_days=config["time_limit"] or NO_LIMIT, **overrides)
    equity, trades, state = simulate(stocks, calendar, signals, rules, config["capital"], CostModel(),
                                     return_state=True, **extra)
    notes = [f"Rules in use: calmest shares + {', '.join(features) or 'nothing extra'}."]
    if "exposure" in extra:
        share = extra["exposure"](latest)
        notes.append(f"Money conditions: invest in {round(config['top_k'] * share)} of {config['top_k']} slots "
                     f"({share:.0%}), based on NRB data published so far.")
    days_since_refresh = len(calendar[(calendar > dates[-1]) & (calendar <= latest)])
    text = report(config, stocks, index_close, equity, trades, state, latest,
                  config["refresh_every"] - days_since_refresh, notes)

    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / f"{latest:%Y-%m-%d}.md").write_text(text + "\n")
    trades.to_csv(out / "trades.csv", index=False, date_format="%Y-%m-%d")
    print(text)


if __name__ == "__main__":
    main()
