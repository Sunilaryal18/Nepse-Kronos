"""Prices for every NEPSE company, including ones that later merged or were delisted.

Listed companies keep the existing adjusted series (socrateai-official/nepse-open-data), which already
handles bonus, right, cash-dividend and merger adjustments. Companies missing from it (merged or delisted)
come from rajeevpaudel/nepse-history (MIT), back-adjusted here with the ShareSansar corporate actions
(cash included, to match), and any remaining one-day drop beyond the ±10% price limit is treated as an
unrecorded corporate action and repaired.

Why not rebuild everything from raw prices? Checked 2026-10-04: our corporate-action data lacks merger
swap ratios, so half the listed series drifted from the reference by >5% somewhere.

NEPSE adjusts the price on the book-closure date itself (checked on NABIL 2023-01-02 and NICA 2023-10-03):
adjusted = (previous close - cash + right_ratio * right_price) / (1 + bonus + right_ratio).

Usage:
    python -m nepse_kronos.history
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nepse_kronos.schema import CANONICAL_COLUMNS, PRICE_COLUMNS, normalize_ohlcv

NOT_SHARES = {"Mutual Fund", "Corporate Debentures", "Promoter Share", "Preference", "Govt Bonds"}
FACE_VALUE = 100.0  # cash dividends and right prices are quoted against Rs 100 face value
RAW_DIR = "data/nepse/source/nepse-history/data/ohlcv"
EVENT_COLUMNS = ["symbol", "book_close", "cash", "bonus_pct", "right_ratio", "right_price"]


def load_raw_ohlcv(ohlcv_dir):
    """{symbol: DataFrame indexed by date} from one-file-per-day CSVs."""
    frames = [pd.read_csv(p) for p in sorted(Path(ohlcv_dir).rglob("*.csv"))]
    rows = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    rows = rows.rename(columns={"date": "timestamps", "turnover": "amount"})
    rows["timestamps"] = pd.to_datetime(rows["timestamps"])
    rows = rows.drop_duplicates(["symbol", "timestamps"], keep="last").sort_values(["symbol", "timestamps"])
    return {symbol: g.set_index("timestamps")[["open", "high", "low", "close", "volume", "amount"]]
            for symbol, g in rows.groupby("symbol")}


def corporate_events(actions, rights):
    """One row per (symbol, book closure) with cash (Rs per share), bonus %, right ratio and right price."""
    div = actions.rename(columns={"cash_pct": "cash"})[["symbol", "book_close", "cash", "bonus_pct"]]
    rgt = rights.rename(columns={"ratio_new_per_old": "right_ratio"})[["symbol", "book_close", "right_ratio"]]
    events = pd.concat([div, rgt], ignore_index=True)
    events["book_close"] = pd.to_datetime(events["book_close"], errors="coerce")
    events = events.dropna(subset=["book_close"])
    for col in ["cash", "bonus_pct", "right_ratio"]:
        events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0.0)
    events = events.groupby(["symbol", "book_close"], as_index=False)[["cash", "bonus_pct", "right_ratio"]].sum()
    events = events[(events[["cash", "bonus_pct", "right_ratio"]] > 0).any(axis=1)]
    return events.assign(cash=events["cash"] * FACE_VALUE / 100, right_price=FACE_VALUE)[EVENT_COLUMNS]


def _price_after(before, cash, ev):
    right = ev.right_ratio or 0.0
    return (before - cash + right * ev.right_price) / (1 + ev.bonus_pct / 100 + right)


def _scale(df, factors):
    out = df.copy()
    out[PRICE_COLUMNS] = out[PRICE_COLUMNS].mul(factors, axis=0)
    out["volume"] = out["volume"] / factors
    return out


def adjustment_factors(df, events, include_cash=False):
    """Multiplier for each row's prices so that history is comparable with today's shares."""
    factors = np.ones(len(df))
    for ev in events.sort_values("book_close").itertuples():
        pos = df.index.searchsorted(pd.Timestamp(ev.book_close))  # first trading day on/after book closure
        if pos == 0 or pos >= len(df):
            continue
        before = df["close"].iloc[pos - 1]
        after = _price_after(before, ev.cash if include_cash else 0.0, ev)
        if before > 0 and after > 0:
            factors[:pos] *= after / before
    return pd.Series(factors, index=df.index)


def adjust_prices(df, events, include_cash=False):
    return _scale(df, adjustment_factors(df, events, include_cash))


def cash_per_unit(df, events):
    """Cash dividend per unit of the bonus/right-adjusted series, keyed by the ex-date (first trading day)."""
    factors = adjustment_factors(df, events, include_cash=False)
    paid = {}
    for ev in events.itertuples():
        pos = df.index.searchsorted(pd.Timestamp(ev.book_close))
        if ev.cash > 0 and 0 < pos < len(df):
            paid[df.index[pos]] = paid.get(df.index[pos], 0.0) + ev.cash * factors.iloc[pos - 1]
    return pd.Series(paid, dtype=float)


def file_name(symbol):
    """Some debenture symbols contain '/', which can't be in a file name."""
    return f"{symbol.replace('/', '_')}.csv"


def fix_stale_adjustments(reference, official_close, events, min_effect=0.02):
    """Apply corporate actions that the reference adjusted series has not applied yet.

    Around each book closure, compare the reference with the official unadjusted close: if the reference
    already applied the action, their ratio jumps by the expected factor; if it is missing, the ratio
    doesn't move. Missing actions are applied to all earlier rows (cash included, as the reference is
    total-return). Actions smaller than `min_effect` are skipped (too small to tell apart).
    Returns (fixed series, list of ex-dates applied).
    """
    fixed = reference.copy()
    official_close = official_close.dropna()
    ratio = fixed["close"] / official_close.reindex(fixed.index)
    applied = []
    for ev in events.sort_values("book_close").itertuples():
        pos = fixed.index.searchsorted(pd.Timestamp(ev.book_close))
        if pos == 0 or pos >= len(fixed):
            continue
        before, after = ratio.iloc[pos - 1], ratio.iloc[pos]
        prior = official_close.loc[:fixed.index[pos - 1]]
        if pd.isna(before) or pd.isna(after) or prior.empty:
            continue
        price = prior.iloc[-1]
        factor = _price_after(price, ev.cash, ev) / price
        if not 0 < factor < 1 - min_effect:
            continue
        move = before / after
        if abs(move - 1) < abs(move - factor):          # the reference didn't adjust here
            fixed.iloc[:pos, fixed.columns.get_indexer(PRICE_COLUMNS)] *= factor
            fixed.iloc[:pos, fixed.columns.get_loc("volume")] /= factor
            ratio.iloc[:pos] *= factor
            applied.append(fixed.index[pos])
    return fixed, applied


def repair_gaps(df, limit=0.12):
    """Undo one-day drops larger than NEPSE's ±10% price limit: they are corporate actions we have no record of."""
    close = df["close"].to_numpy(dtype=float)
    factors = np.ones(len(df))
    for k in range(1, len(df)):
        change = close[k] / close[k - 1] - 1 if close[k - 1] > 0 else 0.0
        if change < -limit:
            factors[:k] *= close[k] / close[k - 1]
    return _scale(df, factors)


def load_official_close(unadjusted_dir):
    """{symbol: official unadjusted close} from socrateai's one-file-per-day folder."""
    frames = [pd.read_csv(p, dtype=str, usecols=lambda c: c in ("date", "symbol", "close"))
              for p in sorted(Path(unadjusted_dir).glob("*.csv"))]
    rows = pd.concat(frames, ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    rows = rows.dropna().drop_duplicates(["symbol", "date"], keep="last")
    return {s: g.set_index("date")["close"].sort_index() for s, g in rows.groupby("symbol")}


def _with_live_days(series, live_dir, symbol):
    """Append days from data/nepse/live that are newer than the series (NEPSE website, official prices)."""
    path = Path(live_dir) / f"{symbol}.csv" if live_dir else None
    if path is None or not path.exists():
        return series, pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    live = pd.read_csv(path, parse_dates=["timestamps"], index_col="timestamps")
    live = live[live.index > series.index.max()][series.columns]
    return pd.concat([series, live]), live["close"]


def build(raw, events, reference_dir, out_dir, official=None, live_dir=None):
    """Listed companies: keep the reference adjusted series. Others (merged/delisted): adjust raw prices.

    Both end up as total-return series (cash dividends included), like the reference data. With
    `official` unadjusted closes, corporate actions the reference hasn't applied yet are applied.
    With `live_dir`, newer days fetched from the NEPSE website are appended to listed series and indices.
    Returns (copied, added, repaired_gaps, stale_fixes).
    """
    reference_dir, out_dir = Path(reference_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = {p.stem for p in reference_dir.glob("*.csv")}
    by_symbol = dict(tuple(events.groupby("symbol")))
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    stale_fixes = 0
    for symbol in reference:
        series = pd.read_csv(reference_dir / f"{symbol}.csv", parse_dates=["timestamps"], index_col="timestamps")
        series, live_close = _with_live_days(series, live_dir, symbol)
        if official and symbol in official and symbol in by_symbol:
            closes = pd.concat([official[symbol], live_close[live_close.index > official[symbol].index.max()]])
            series, applied = fix_stale_adjustments(series, closes, by_symbol[symbol])
            stale_fixes += len(applied)
        series.reset_index().to_csv(out_dir / f"{symbol}.csv", index=False, date_format="%Y-%m-%d")
    added = repaired = 0
    for symbol, df in raw.items():
        if symbol in reference or file_name(symbol)[:-4] in reference:
            continue
        adjusted = adjust_prices(df, by_symbol.get(symbol, empty), include_cash=True)
        fixed = repair_gaps(adjusted)
        repaired += int((fixed["close"] / adjusted["close"]).round(6).diff().fillna(0).ne(0).sum())
        normalize_ohlcv(fixed.reset_index())[CANONICAL_COLUMNS].to_csv(out_dir / file_name(symbol), index=False, date_format="%Y-%m-%d")
        added += 1
    return len(reference), added, repaired, stale_fixes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--meta-dir", default="data/nepse/meta")
    parser.add_argument("--reference-dir", default="data/nepse/clean", help="existing adjusted data for listed companies")
    parser.add_argument("--out-dir", default="data/nepse/clean_full")
    parser.add_argument("--unadjusted-dir", default="data/nepse/source/nepse-open-data/ohlc_unadjusted_stock")
    parser.add_argument("--live-dir", default="data/nepse/live", help="newer days from the NEPSE website")
    args = parser.parse_args(argv)

    meta = Path(args.meta_dir)
    events = corporate_events(pd.read_csv(meta / "corporate_actions.csv"), pd.read_csv(meta / "rights.csv"))
    official = load_official_close(args.unadjusted_dir) if Path(args.unadjusted_dir).exists() else None
    if official:
        from nepse_kronos.benchmark import market_dividend_yield
        sectors = pd.read_csv(meta / "sectors.csv") if (meta / "sectors.csv").exists() else pd.DataFrame(columns=["symbol", "sector"])
        not_shares = set(sectors.loc[sectors["sector"].isin(NOT_SHARES), "symbol"])
        market_dividend_yield(events, official, exclude=not_shares).rename_axis("year").to_csv(
            meta / "market_dividend_yield.csv")
    copied, added, repaired, fixes = build(load_raw_ohlcv(args.raw_dir), events, args.reference_dir, args.out_dir,
                                           official, args.live_dir)
    print(f"{args.out_dir}: kept {copied} listed series ({fixes} missed adjustments applied), "
          f"added {added} merged/delisted series, repaired {repaired} unexplained price gaps")


if __name__ == "__main__":
    main()
