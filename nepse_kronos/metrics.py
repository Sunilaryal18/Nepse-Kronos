import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 240  # ~5 sessions a week minus NEPSE holidays


def _stats(values):
    returns = values.pct_change().dropna()
    years = len(returns) / TRADING_DAYS_PER_YEAR
    total = values.iloc[-1] / values.iloc[0] - 1.0
    std = returns.std()
    return {
        "total_return": total,
        "cagr": (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan,
        "volatility": std * np.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe": returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else np.nan,
        "max_drawdown": (values / values.cummax() - 1.0).min(),
    }


def performance(equity, benchmark):
    benchmark = benchmark.reindex(equity.index).ffill()
    return {"strategy": _stats(equity), "benchmark": _stats(benchmark)}


def trade_summary(trades, equity):
    years = max(len(equity) - 1, 1) / TRADING_DAYS_PER_YEAR
    bought = trades.loc[trades["side"] == "buy", "value"].sum()
    paid = trades[trades["side"] == "dividend"]
    sold = trades.loc[trades["side"] == "sell", "reason"].value_counts()
    return {
        "trades": int(trades["side"].isin(["buy", "sell"]).sum()),
        "fees": trades["fees"].sum(),
        "tax": trades["tax"].sum(),
        "dividends": (paid["value"] - paid["tax"]).sum(),
        "annual_turnover": bought / equity.mean() / years,
        **{f"sold_{reason}": int(sold.get(reason, 0)) for reason in ("rank", "stop_loss", "time_limit")},
    }


def rank_ic(signals, panel, calendar, horizon):
    """Per signal date: rank correlation between predicted and actual `horizon`-day returns."""
    ics = {}
    for date, signal in signals.items():
        later = calendar[calendar > date]
        if len(later) < horizon:
            continue
        end = later[horizon - 1]
        actual = {}
        for symbol in signal.index:
            df = panel[symbol]
            if date in df.index:
                actual[symbol] = df.loc[:end, "close"].iloc[-1] / df.at[date, "close"] - 1.0
        actual = pd.Series(actual, dtype=float)
        if len(actual) >= 5:
            ics[date] = signal.reindex(actual.index).rank().corr(actual.rank())
    return pd.Series(ics, name="rank_ic", dtype=float)


def summarize_ic(ic):
    n = len(ic)
    std = ic.std()
    return {
        "periods": n,
        "mean_ic": ic.mean(),
        "ic_t_stat": ic.mean() / std * np.sqrt(n) if n > 1 and std > 0 else np.nan,
        "share_positive": (ic > 0).mean(),
    }


def yearly_table(equity, benchmark):
    """Return per calendar year for the strategy and the benchmark."""
    both = pd.DataFrame({"strategy": equity, "benchmark": benchmark.reindex(equity.index).ffill()})
    year_end = both.groupby(both.index.year).last()
    year_start = year_end.shift(1)
    year_start.iloc[0] = both.iloc[0]
    return year_end / year_start - 1.0
