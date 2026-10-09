"""Track the shares you actually own: value, gain or loss, today's move, take-home if sold, and a health check.

Holdings live in data/nepse/my_holdings.csv (git-ignored, never published):
    symbol,units,buy_price,note

Usage:
    python -m nepse_kronos.my_portfolio
"""
import argparse
from pathlib import Path

import pandas as pd

from nepse_kronos.costs import CostModel
from nepse_kronos.investor_backtest import load_sectors, tradable_stocks
from nepse_kronos.picks import _profit_growth, technicals
from nepse_kronos.universe import load_panel

HOLDINGS = Path("data/nepse/my_holdings.csv")
COLUMNS = ["symbol", "sector", "units", "buy_price", "last", "last_date", "cost", "value", "gain", "gain_pct",
           "day_change", "note"]


def load_holdings(path=HOLDINGS):
    holdings = pd.read_csv(path, dtype={"note": str}).fillna({"note": ""})
    holdings["symbol"] = holdings["symbol"].str.strip().str.upper()
    return holdings


def value_holdings(holdings, stocks, day, sectors):
    """One row per holding, valued at the latest close on or before `day` (unknown symbols kept, not valued)."""
    rows = []
    for h in holdings.itertuples():
        row = {"symbol": h.symbol, "sector": sectors.get(h.symbol, ""), "units": h.units, "buy_price": h.buy_price,
               "cost": h.units * h.buy_price, "note": h.note}
        if h.symbol in stocks:
            close = stocks[h.symbol]["close"].loc[:day]
            row.update(last=close.iloc[-1], last_date=close.index[-1], value=h.units * close.iloc[-1],
                       day_change=close.iloc[-1] / close.iloc[-2] - 1 if len(close) > 1 else 0.0)
            row.update(gain=row["value"] - row["cost"], gain_pct=close.iloc[-1] / h.buy_price - 1)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(valued, costs):
    """Totals, today's move in rupees, and what selling everything today would leave after fees and tax."""
    priced = valued.dropna(subset=["value"])
    previous = (priced["value"] / (1 + priced["day_change"])).sum()
    fees = priced["value"].map(costs.trade_cost)
    tax = [costs.capital_gains_tax(v - f, c) for v, f, c in zip(priced["value"], fees, priced["cost"])]
    value, cost = priced["value"].sum(), priced["cost"].sum()
    return {"cost": cost, "value": value, "gain": value - cost, "gain_pct": value / cost - 1 if cost else 0.0,
            "today_rs": value - previous, "today_pct": value / previous - 1 if previous else 0.0,
            "take_home": value - fees.sum() - sum(tax), "unpriced": valued.loc[valued["value"].isna(), "symbol"].tolist()}


def health_notes(symbol, stocks, reports, day):
    """Plain-English checks: trend and RSI from the chart, valuation and profit growth from published reports."""
    if symbol not in stocks:
        return ["No price data for this symbol."]
    chart = technicals(stocks[symbol], day)
    notes = [{"up": "Uptrend: price above its rising 50-day average.",
              "down": "Downtrend: price below its falling 50-day average.",
              "sideways": "No clear trend around its 50-day average."}[chart["trend"]]]
    if chart["rsi14"] > 70:
        notes.append(f"Overheated (RSI {chart['rsi14']:.0f}).")
    elif chart["rsi14"] < 30:
        notes.append(f"Oversold (RSI {chart['rsi14']:.0f}) — often near a short-term low.")
    notes.append(f"At {chart['range52']:.0%} of its 52-week range.")

    reports = reports.assign(submitted=pd.to_datetime(reports["submitted"])).sort_values("submitted")
    profit = _profit_growth(reports, symbol, day)
    price = stocks[symbol]["close"].loc[:day].iloc[-1]
    if profit is None:
        published = reports[(reports["symbol"] == symbol) & (reports["submitted"] < day)]
        if published.empty:
            notes.append("No published profit report found.")
            return notes
        eps, book, growth = published.iloc[-1]["eps"], published.iloc[-1]["net_worth_per_share"], None
    else:
        eps, eps_before, book = profit
        growth = eps / eps_before - 1 if eps_before > 0 else None
    if eps <= 0:
        notes.append("Making a loss in its latest report.")
    else:
        pe = price / eps
        level = "reasonable" if pe <= 25 else "high" if pe <= 40 else "expensive"
        notes.append(f"Priced at {pe:.0f}× profit ({level})" + (f"; profit {growth:+.0%} on a year ago." if growth is not None else "."))
    if book and book > 0 and price / book > 5:
        notes.append(f"Price is {price / book:.1f}× book value (expensive).")
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdings", default=str(HOLDINGS))
    parser.add_argument("--clean-dir", default="data/nepse/clean_full")
    parser.add_argument("--meta-dir", default="data/nepse/meta")
    args = parser.parse_args(argv)

    sectors = load_sectors(args.meta_dir)
    panel = load_panel(args.clean_dir, include_indices=True)
    stocks, index = tradable_stocks(panel, sectors), panel["NEPSE_INDEX"]["close"]
    day = index.index[-1]
    valued = value_holdings(load_holdings(args.holdings), stocks, day, sectors)
    total = summarize(valued, CostModel())
    reports = pd.read_csv(Path(args.meta_dir) / "fundamentals.csv")

    print(f"My portfolio — {day:%Y-%m-%d} (NEPSE index today {index.iloc[-1] / index.iloc[-2] - 1:+.2%})\n")
    for r in valued.itertuples():
        if pd.isna(r.value):
            print(f"{r.symbol:6} {r.units:>5} @ {r.buy_price:>7.1f}  — no price data")
            continue
        print(f"{r.symbol:6} {r.units:>5} @ {r.buy_price:>7.1f} → {r.last:>7.1f}  value Rs {r.value:>10,.0f}  "
              f"{r.gain:>+10,.0f} ({r.gain_pct:+.1%})  today {r.day_change:+.1%}" + (f"  [{r.note}]" if r.note else ""))
        for note in health_notes(r.symbol, stocks, reports, day):
            print(f"         · {note}")
    print(f"\nTotal: cost Rs {total['cost']:,.0f} → value Rs {total['value']:,.0f}  {total['gain']:+,.0f} ({total['gain_pct']:+.1%})"
          f"  | today {total['today_rs']:+,.0f} ({total['today_pct']:+.2%})")
    print(f"If you sold everything today: about Rs {total['take_home']:,.0f} after broker fees and 10% tax on profits.")


if __name__ == "__main__":
    main()
