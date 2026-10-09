"""Backtest the calm-shares system with the "top investor" features switched on one at a time.

Base system: rank the most-traded shares from calmest to jumpiest every 10 trading days, hold up to 20,
keep a share while it stays in the top 50, no automatic stop-loss, real NEPSE fees and tax.

Features (comma-separated in --features):
    money      invest 50-100% depending on interest rates and bank lending room (NRB data)
    quality    skip shares with a published loss, P/E > 40 or price > 5x book value
    supply     skip shares shortly before promoter unlocks or right-share book closures
    broker     prefer shares big brokers are buying; skip possible pumping
    sectors    at most 5 shares from one sector
    taxwait    hold a profitable share a little longer to reach the lower 1-year tax rate
    interest   idle cash earns the banks' average deposit rate (after 6% tax)

Usage:
    python -m nepse_kronos.investor_backtest --features money,quality --start 2017-06-01
"""
import argparse
from pathlib import Path

import pandas as pd

from nepse_kronos.benchmark import total_return_index
from nepse_kronos.costs import CostModel
from nepse_kronos.history import NOT_SHARES
from nepse_kronos.investor import (broker_tiebreak, deposit_rate, money_exposure, pump_blocklist,
                                   quality_blocklist, supply_blocklist)
from nepse_kronos.metrics import performance, trade_summary, yearly_table
from nepse_kronos.portfolio import Rules, simulate
from nepse_kronos.signals import cached_signal, lowvol_signal
from nepse_kronos.universe import liquid_universe, load_panel, rebalance_dates

FEATURES = ["money", "quality", "supply", "broker", "sectors", "taxwait", "interest"]
NOT_TRADED = NOT_SHARES
NO_LIMIT = 10**6


def parse_features(parser, text):
    features = [f for f in text.split(",") if f]
    unknown = set(features) - set(FEATURES)
    if unknown:
        parser.error(f"unknown features: {sorted(unknown)}")
    return features


def lowvol_signals(stocks, dates, cache, universe, min_history, window):
    """Calmest-first ranking of the most-traded shares on each date, cached on disk."""
    signals = {}
    for date in dates:
        symbols = liquid_universe(stocks, date, top_n=universe, min_history=min_history)
        signals[date] = cached_signal(cache, date, lambda: lowvol_signal(stocks, symbols, date, window))
    return signals


def load_sectors(meta):
    path = Path(meta) / "sectors.csv"
    return dict(pd.read_csv(path)[["symbol", "sector"]].values) if path.exists() else {}


def tradable_stocks(panel, sectors):
    """Shares only: drop indices, mutual funds, debentures, promoter and preference shares."""
    return {s: df for s, df in panel.items() if not s.endswith("_INDEX") and sectors.get(s) not in NOT_TRADED}


def apply_features(features, stocks, signals, meta, sectors, money_months=12, max_per_sector=5):
    """Turn feature names into (adjusted signals, extra simulate() arguments, Rules overrides)."""
    meta = Path(meta)
    blocklists, extra, overrides = [], {}, {}
    if "broker" in features:
        broker = pd.read_csv(meta / "broker_features.csv.gz", parse_dates=["date"])
        signals = {d: broker_tiebreak(s, broker, d) for d, s in signals.items()}
        blocklists.append(pump_blocklist(broker, stocks))
    if "quality" in features:
        blocklists.append(quality_blocklist(pd.read_csv(meta / "fundamentals.csv"), stocks))
    if "supply" in features:
        blocklists.append(supply_blocklist(pd.read_csv(meta / "lockins.csv"), pd.read_csv(meta / "rights.csv")))
    if blocklists:
        extra["blocklist"] = lambda day: set().union(*(b(day) for b in blocklists))
    if "money" in features:
        extra["exposure"] = money_exposure(pd.read_csv(meta / "macro_monthly.csv",
                                                       parse_dates=["month", "available_from"]), months=money_months)
    if "interest" in features:
        extra["cash_rate"] = deposit_rate(pd.read_csv(meta / "macro_monthly.csv", parse_dates=["available_from"]))
    if "sectors" in features:
        extra["sectors"] = sectors
        overrides["max_per_sector"] = max_per_sector
    if "taxwait" in features:
        overrides["tax_wait_days"] = 60
    return signals, extra, overrides


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", default="data/nepse/clean_full")
    parser.add_argument("--meta-dir", default="data/nepse/meta")
    parser.add_argument("--out-dir", default="outputs/nepse/investor")
    parser.add_argument("--features", default="", help=",".join(FEATURES))
    parser.add_argument("--start", default="2017-06-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--benchmark", default="NEPSE_INDEX")
    parser.add_argument("--universe", type=int, default=150)
    parser.add_argument("--min-history", type=int, default=128)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--refresh-every", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--keep-rank", type=int, default=50)
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--money-months", type=int, default=12)
    parser.add_argument("--max-per-sector", type=int, default=5)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    features = parse_features(parser, args.features)
    meta = Path(args.meta_dir)

    panel = load_panel(args.clean_dir, include_indices=True)
    index_close = panel[args.benchmark]["close"]
    sectors = load_sectors(meta)
    stocks = tradable_stocks(panel, sectors)
    end = pd.Timestamp(args.end) if args.end else index_close.index[-1]
    calendar = index_close.index[index_close.index <= end]
    dates = rebalance_dates(calendar, args.start, end, args.refresh_every)

    cache = Path(args.out_dir) / "signals" / f"{Path(args.clean_dir).name}-lowvol-w{args.window}-u{args.universe}"
    signals = lowvol_signals(stocks, dates, cache, args.universe, args.min_history, args.window)
    signals, extra, overrides = apply_features(features, stocks, signals, meta, sectors,
                                               args.money_months, args.max_per_sector)
    rules = Rules(top_k=args.top_k, keep_rank=args.keep_rank, stop_loss=1.0, time_stop_days=NO_LIMIT, **overrides)
    # Prices already include cash dividends (total-return adjusted), so the engine pays none separately.
    equity, trades = simulate(stocks, calendar, signals, rules, args.capital, CostModel(), **extra)
    price_index = index_close.reindex(equity.index).ffill()
    yields_file = meta / "market_dividend_yield.csv"
    with_dividends = yields_file.exists()   # fair yardstick: our prices include dividends, the index doesn't
    if with_dividends:
        yields = pd.read_csv(yields_file, index_col="year")["dividend_yield"]
        benchmark = total_return_index(price_index, yields)
    else:
        benchmark = price_index
    benchmark = benchmark / benchmark.iloc[0] * args.capital
    print(f"benchmark: NEPSE index{' + estimated dividends' if with_dividends else ''} "
          f"(price index alone: {price_index.iloc[-1] / price_index.iloc[0] - 1:+.1%})")

    print(f"features: {','.join(features) or 'none'} | {Path(args.clean_dir).name} | "
          f"{equity.index[0]:%Y-%m-%d} -> {equity.index[-1]:%Y-%m-%d}")
    print(pd.DataFrame(performance(equity, benchmark)).T.round(4).to_string())
    print(yearly_table(equity, benchmark).round(4).to_string())
    for key, value in trade_summary(trades, equity).items():
        print(f"{key:>16}: {value:,.2f}" if isinstance(value, float) else f"{key:>16}: {value}")

    out = Path(args.out_dir) / (args.tag or ("-".join(features) or "base"))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": equity, "benchmark": benchmark}).rename_axis("timestamps").to_csv(
        out / "equity.csv", date_format="%Y-%m-%d")
    trades.to_csv(out / "trades.csv", index=False, date_format="%Y-%m-%d")


if __name__ == "__main__":
    main()
