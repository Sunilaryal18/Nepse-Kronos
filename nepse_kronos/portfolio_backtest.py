"""Would a rules-based NEPSE portfolio beat holding the index after real fees and tax?

Every trading day: check each holding against the selling rules (loss limit, time limit, keep zone)
and fill empty slots with the best-ranked shares. The ranking is refreshed every --refresh-every days.

Usage:
    python -m nepse_kronos.portfolio_backtest --signal momentum --start 2017-06-01
    python -m nepse_kronos.portfolio_backtest --signal kronos --start 2024-10-01      # ~2 h, resumable
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nepse_kronos.costs import CostModel
from nepse_kronos.metrics import performance, rank_ic, summarize_ic, trade_summary, yearly_table
from nepse_kronos.portfolio import Rules, simulate
from nepse_kronos.signals import cached_signal, kronos_signal, lowvol_signal, momentum_signal, steady_signal
from nepse_kronos.trading_calendar import load_holidays, next_trading_days
from nepse_kronos.universe import liquid_universe, load_panel, rebalance_dates

SIMPLE = {"momentum": momentum_signal, "steady": steady_signal, "lowvol": lowvol_signal}


def future_dates(calendar, date, horizon):
    """The next `horizon` market days after date; beyond the data, continue with the NEPSE calendar."""
    later = list(calendar[calendar > date][:horizon])
    if len(later) < horizon:
        last = later[-1] if later else date
        later += list(next_trading_days(last, horizon - len(later), load_holidays()))
    return pd.DatetimeIndex(later)


def signal_tag(args):
    if args.signal in SIMPLE:
        return f"{args.signal}-w{args.window}"
    return f"kronos-{Path(args.model).name}-lb{args.lookback}-h{args.horizon}-s{args.sample_count}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", default="data/nepse/clean")
    parser.add_argument("--out-dir", default="outputs/nepse/portfolio")
    parser.add_argument("--signal", choices=["kronos", *SIMPLE], default="momentum")
    parser.add_argument("--benchmark", default="NEPSE_INDEX")
    parser.add_argument("--start", default="2024-10-01")
    parser.add_argument("--end", default=None, help="Last date (default: latest data)")
    parser.add_argument("--refresh-every", type=int, default=10, help="Market days between ranking refreshes")
    parser.add_argument("--horizon", type=int, default=10, help="Kronos forecast / ranking-check horizon in days")
    parser.add_argument("--window", type=int, default=60, help="Look-back days for the simple signals")
    parser.add_argument("--lookback", type=int, default=128, help="Kronos context; also the minimum history")
    parser.add_argument("--universe", type=int, default=150, help="Number of most-traded shares to rank")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--keep-rank", type=int, default=50)
    parser.add_argument("--stop-loss", type=float, default=0.15)
    parser.add_argument("--time-stop", type=int, default=60)
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="NeoQuasar/Kronos-small")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None, help="Name for the results folder")
    args = parser.parse_args(argv)

    panel = load_panel(args.clean_dir, include_indices=True)
    index_close = panel[args.benchmark]["close"]
    stocks = {s: df for s, df in panel.items() if not s.endswith("_INDEX")}
    end = pd.Timestamp(args.end) if args.end else index_close.index[-1]
    calendar = index_close.index[index_close.index <= end]
    dates = rebalance_dates(calendar, args.start, end, args.refresh_every)

    cache_dir = Path(args.out_dir) / "signals" / signal_tag(args)
    predictor = None
    if args.signal == "kronos":
        from nepse_kronos.forecast import load_predictor
        predictor = load_predictor(args.model, args.tokenizer, device=args.device)

    signals = {}
    for i, date in enumerate(dates, 1):
        symbols = liquid_universe(stocks, date, top_n=args.universe, min_history=args.lookback)
        if args.signal == "kronos":
            compute = lambda: kronos_signal(predictor, stocks, symbols, date, future_dates(calendar, date, args.horizon),
                                            args.lookback, args.sample_count, args.batch_size)
        else:
            compute = lambda: SIMPLE[args.signal](stocks, symbols, date, args.window)
        signals[date] = cached_signal(cache_dir, date, compute)
        print(f"[{i}/{len(dates)}] {date:%Y-%m-%d}: ranked {len(signals[date])} shares", flush=True)

    rules = Rules(top_k=args.top_k, keep_rank=args.keep_rank, stop_loss=args.stop_loss, time_stop_days=args.time_stop)
    equity, trades = simulate(stocks, calendar, signals, rules, args.capital, CostModel())
    benchmark = index_close.reindex(equity.index).ffill()
    benchmark = benchmark / benchmark.iloc[0] * args.capital
    ic = rank_ic(signals, stocks, calendar, args.horizon)

    print()
    print(pd.DataFrame(performance(equity, benchmark)).T.round(4).to_string())
    print()
    print(yearly_table(equity, benchmark).round(4).to_string())
    print()
    for key, value in {**trade_summary(trades, equity), **summarize_ic(ic)}.items():
        print(f"{key:>16}: {value:,.4f}" if isinstance(value, float) else f"{key:>16}: {value}")

    out = Path(args.out_dir) / (args.tag or f"{signal_tag(args)}-from{pd.Timestamp(args.start):%Y-%m-%d}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": equity, "benchmark": benchmark}).rename_axis("timestamps").to_csv(
        out / "equity.csv", date_format="%Y-%m-%d")
    trades.to_csv(out / "trades.csv", index=False, date_format="%Y-%m-%d")
    ic.rename_axis("timestamps").to_csv(out / "ic.csv", date_format="%Y-%m-%d")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity, label=f"Rules + {args.signal} ranking, after fees and tax", color="tab:red")
    ax.plot(benchmark.index, benchmark, label=f"{args.benchmark} (just hold)", color="tab:blue")
    ax.set_ylabel("Portfolio value (NPR)")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
