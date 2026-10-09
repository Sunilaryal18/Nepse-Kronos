"""What experienced Nepali investors look at, turned into rules the trading engine can use.

Every rule only uses information that was public on the decision day.
"""
from functools import cache

import numpy as np
import pandas as pd


def quality_blocklist(reports, panel, max_pe=40.0, max_price_to_book=5.0):
    """Skip shares whose latest published quarterly report shows a loss or a very high price.

    reports: symbol, eps (annualised), net_worth_per_share, submitted. A report counts from the day
    after it was submitted. Shares without any published report are not judged.
    """
    reports = reports.dropna(subset=["submitted"]).assign(submitted=lambda d: pd.to_datetime(d["submitted"]))
    by_symbol = {s: g.sort_values("submitted") for s, g in reports.groupby("symbol") if s in panel}

    def blocked(day):
        out = set()
        for symbol, rows in by_symbol.items():
            k = rows["submitted"].searchsorted(day, side="left") - 1
            if k < 0:
                continue
            latest = rows.iloc[k]
            close = panel[symbol].loc[:day, "close"]
            if close.empty:
                continue
            price, eps, book = close.iloc[-1], latest["eps"], latest["net_worth_per_share"]
            if pd.notna(eps) and (eps <= 0 or price / eps > max_pe):
                out.add(symbol)
            elif pd.notna(book) and book > 0 and price / book > max_price_to_book:
                out.add(symbol)
        return out
    return cache(blocked)


def supply_blocklist(lockins, rights, lock_days=60, right_days=30):
    """Skip shares shortly before promoter shares unlock or a right-share book closure (new supply)."""
    ends = [(s, pd.Timestamp(d)) for s, d in zip(lockins["symbol"], lockins["promoter_lock_end"]) if pd.notna(d)]
    closes = [(s, pd.Timestamp(d)) for s, d in zip(rights["symbol"], rights["book_close"]) if pd.notna(d)]

    def blocked(day):
        out = {s for s, end in ends if 0 < (end - day).days <= lock_days}
        out |= {s for s, bc in closes if 0 < (bc - day).days <= right_days}
        return out
    return cache(blocked)


def pump_blocklist(features, panel, hhi_quantile=0.95, jump=0.20, window=20):
    """Skip shares where a few brokers dominate buying while the price shot up (possible pumping)."""
    features = features.assign(date=lambda d: pd.to_datetime(d["date"]))
    by_date = {d: g.set_index("symbol")["buyer_hhi"] for d, g in features.groupby("date")}
    dates = pd.DatetimeIndex(sorted(by_date))

    def blocked(day):
        k = dates.searchsorted(day, side="right") - 1
        if k < 0:
            return set()
        hhi = by_date[dates[k]].dropna()
        if hhi.empty:
            return set()
        crowded = hhi[hhi >= hhi.quantile(hhi_quantile)].index
        out = set()
        for symbol in crowded:
            if symbol not in panel:
                continue
            close = panel[symbol].loc[:day, "close"]
            if len(close) > window and close.iloc[-1] / close.iloc[-1 - window] - 1 > jump:
                out.add(symbol)
        return out
    return cache(blocked)


def money_exposure(macro, low=0.5, months=12):
    """Share of the portfolio slots to fill, from published NRB data (1.0 before data exists).

    Score +1 each: interbank rate below its previous-12-month average, credit-to-deposit ratio below
    its previous-`months` average, lending rate lower than 3 months earlier. 0 -> `low`, 3 -> 1.0.
    """
    macro = macro.sort_values("available_from").reset_index(drop=True)
    available = pd.to_datetime(macro["available_from"])

    def exposure(day):
        n = int(available.searchsorted(day, side="right"))
        if n == 0:
            return 1.0
        rows = macro.iloc[:n]
        last, before = rows.iloc[-1], rows.iloc[-1 - months:-1]
        score = 0
        if not before.empty:
            for column in ("interbank", "cd_ratio"):
                if pd.notna(last[column]) and last[column] < before[column].mean():
                    score += 1
        if n > 3 and pd.notna(last["lending_rate"]) and last["lending_rate"] < rows["lending_rate"].iloc[-4]:
            score += 1
        return low + (1.0 - low) * score / 3
    return cache(exposure)


def deposit_rate(macro):
    """Banks' weighted-average deposit rate (as a fraction) from the latest published NRB month; 0 before data."""
    macro = macro.sort_values("available_from").reset_index(drop=True)
    available = pd.to_datetime(macro["available_from"])
    rates = macro["deposit_rate"].ffill() / 100

    def rate(day):
        n = int(available.searchsorted(day, side="right"))
        return 0.0 if n == 0 or pd.isna(rates.iloc[n - 1]) else float(rates.iloc[n - 1])
    return cache(rate)


def broker_tiebreak(signal, features, day, bonus=0.05):
    """Nudge the ranking towards shares the biggest brokers are buying more heavily than usual.

    top5_net_buy is never negative, so "heavy buying" means above that day's median across shares.
    Scores become percentile ranks (0-1) plus/minus `bonus`, so only neighbouring ranks change places.
    """
    dates = pd.to_datetime(features["date"])
    if not (dates <= day).any():
        return signal
    today = features[dates == dates[dates <= day].max()].set_index("symbol")["top5_net_buy"]
    nudge = np.sign(today - today.median()).reindex(signal.index).fillna(0.0) * bonus
    return (signal.rank(pct=True) + nudge).rename(signal.name)
