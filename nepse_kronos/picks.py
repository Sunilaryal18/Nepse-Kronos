"""Shares that make sense to consider buying on a given day, with a buying range and the reasons why.

Screen (fixed before testing; see README section 10):
  business  latest *published* quarterly profit per share up more than 10% on the same quarter a year earlier,
            P/E at most 25, price at most 2.5x book value, not blocked by the quality/new-supply filters
  chart     not in a falling trend (below a falling 50-day average) and not overheated (RSI(14) above 70 or
            up more than 15% in 20 trading days)
  brokers   the biggest brokers not clearly selling
Survivors are ranked calmest first. `track_record` replays the screen on past data.
"""
import numpy as np
import pandas as pd

from nepse_kronos.universe import liquid_universe

COLUMNS = ["symbol", "sector", "price", "buy_low", "buy_high", "eps", "eps_before", "eps_growth", "pe", "pb",
           "volatility", "trend", "rsi14", "ret20", "range52", "vol_ratio", "support", "resistance", "broker_vs_median"]
SECTOR_RISK = {
    "Commercial Bank": "Banks move together with interest rates, NRB rules and bad-loan news.",
    "Development Bank": "Banks move together with interest rates, NRB rules and bad-loan news.",
    "Finance": "Finance companies move with interest rates, NRB rules and bad-loan news.",
    "Microfinance": "Microfinance is sensitive to NRB lending caps and borrower repayment problems.",
    "Hydropower": "Hydropower depends on monsoon output, power-purchase rules and frequent right-share issues.",
    "Life Insurance": "Insurers are sensitive to regulator capital rules and mergers.",
    "Non-Life Insurance": "Insurers are sensitive to regulator capital rules and mergers.",
}


def technicals(df, day):
    """What the candlestick chart says on `day`: trend, RSI(14), recent move, 52-week position, volume, levels."""
    hist = df.loc[:day]
    close = hist["close"]
    ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()
    price = close.iloc[-1]
    slope = ma50.iloc[-1] - ma50.iloc[-21] if len(ma50) > 20 else np.nan
    if price > ma50.iloc[-1] and slope > 0:
        trend = "up"
    elif price < ma50.iloc[-1] and slope < 0:
        trend = "down"
    else:
        trend = "sideways"
    change = close.diff().tail(14)
    gain, loss = change.clip(lower=0).mean(), -change.clip(upper=0).mean()
    rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    year = close.tail(250)
    span = year.max() - year.min()
    return {"ma20": ma20.iloc[-1], "ma50": ma50.iloc[-1], "trend": trend, "rsi14": rsi,
            "ret20": price / close.iloc[-21] - 1 if len(close) > 20 else np.nan,
            "range52": (price - year.min()) / span if span > 0 else 0.5,
            "vol_ratio": hist["volume"].tail(5).mean() / max(hist["volume"].tail(60).median(), 1),
            "support": hist["low"].tail(20).min(), "resistance": hist["high"].tail(20).max()}


def _profit_growth(reports, symbol, day):
    """(latest annualised EPS, EPS of the same quarter a year earlier, book value per share) from published reports."""
    published = reports[(reports["symbol"] == symbol) & (reports["submitted"] < day)]
    if published.empty:
        return None
    latest = published.iloc[-1]
    year_ago = published[(published["quarter"] == latest["quarter"])
                         & (published["fiscal_year"] != latest["fiscal_year"])]
    if year_ago.empty:
        return None
    return latest["eps"], year_ago.iloc[-1]["eps"], latest["net_worth_per_share"]


def daily_picks(stocks, sectors, reports, day, broker=None, blocklist=None, top=5, universe=150, min_history=128,
                min_growth=0.10, max_pe=25.0, max_pb=2.5, chase_limit=0.01, max_rsi=70.0, max_ret20=0.15):
    """Up to `top` shares passing the screen on `day`, calmest first, with a buying range.

    buy_low = the lower of the last close and the 20-day average, but not below the 20-day low (support);
    buy_high = last close + `chase_limit`, but not above the 20-day high (resistance) unless already there.
    """
    day = pd.Timestamp(day)
    reports = reports.assign(submitted=pd.to_datetime(reports["submitted"])).sort_values("submitted")
    blocked = blocklist(day) if blocklist else set()
    broker_today = pd.Series(dtype=float)
    if broker is not None:
        dates = pd.to_datetime(broker["date"])
        if (dates <= day).any():
            today = broker[dates == dates[dates <= day].max()].set_index("symbol")["top5_net_buy"]
            broker_today = today / today.median()
    weak_brokers = set(broker_today[broker_today < 0.5].index)

    rows = []
    for symbol in liquid_universe(stocks, day, top_n=universe, min_history=min_history):
        if symbol in blocked or symbol in weak_brokers:
            continue
        profit = _profit_growth(reports, symbol, day)
        if profit is None:
            continue
        eps, eps_before, book = profit
        if not (eps > 0 and eps_before > 0 and book > 0):
            continue
        close = stocks[symbol]["close"].loc[:day]
        price = close.iloc[-1]
        growth, pe, pb = eps / eps_before - 1, price / eps, price / book
        if growth <= min_growth or pe > max_pe or pb > max_pb:
            continue
        chart = technicals(stocks[symbol], day)
        if chart["trend"] == "down" or chart["rsi14"] > max_rsi or chart["ret20"] > max_ret20:
            continue
        low = max(chart["support"], min(price, chart["ma20"]))
        high = max(price, min(price * (1 + chase_limit), chart["resistance"]))
        rows.append({"symbol": symbol, "sector": sectors.get(symbol, ""), "price": price,
                     "buy_low": round(min(low, price), 1), "buy_high": round(high, 1),
                     "eps": eps, "eps_before": eps_before, "eps_growth": growth, "pe": pe, "pb": pb,
                     "volatility": close.pct_change().tail(60).std(),
                     **{k: chart[k] for k in ("trend", "rsi14", "ret20", "range52", "vol_ratio", "support", "resistance")},
                     "broker_vs_median": broker_today.get(symbol, np.nan)})
    picks = pd.DataFrame(rows, columns=COLUMNS)
    return picks.sort_values(["volatility", "symbol"]).head(top).reset_index(drop=True)


def explain(pick, actions, rights, day, lockins=None):
    """Plain-English reasons to consider the share, and the risks, from the numbers behind the pick."""
    why = [f"Business: profit per share {pick.eps_growth:+.0%} on a year ago (Rs {pick.eps:.1f} vs "
           f"Rs {pick.eps_before:.1f}, annualised), priced at {pick.pe:.1f}× profit and {pick.pb:.2f}× book value "
           "— reasonable for NEPSE."]
    trend = {"up": "uptrend — price above its rising 50-day average",
             "sideways": "sideways — no clear direction around its 50-day average"}[pick.trend]
    why.append(f"Chart: {trend}; not overheated (RSI {pick.rsi14:.0f}, {pick.ret20:+.1%} in 20 days, "
               f"at {pick.range52:.0%} of its 52-week range).")
    why.append(f"Calm: typical daily move {pick.volatility:.1%}.")
    if pd.notna(pick.broker_vs_median) and pick.broker_vs_median > 1:
        why.append(f"Brokers: the biggest buyers are buying {pick.broker_vs_median:.1f}× as heavily as for a typical share.")
    if pick.vol_ratio >= 1.5:
        why.append(f"Volume: trading is {pick.vol_ratio:.1f}× normal — fresh interest; check recent announcements.")

    risks = [SECTOR_RISK.get(pick.sector, "Company-specific news can move the price.")]
    if pick.eps_growth > 1:
        risks.append("Profit more than doubled — this may include one-off gains; don't count on it repeating.")
    own = actions[actions["symbol"] == pick.symbol].assign(book_close=lambda d: pd.to_datetime(d["book_close"]))
    if not own.empty:
        last = own.sort_values("book_close").iloc[-1]
        parts = [f"{last.bonus_pct:g}% bonus" if last.bonus_pct else "", f"{last.cash_pct:g}% cash" if last.cash_pct else ""]
        paid = " + ".join(p for p in parts if p) or "nothing"
        if pd.notna(last.book_close) and last.book_close > day:
            risks.append(f"Book closure on {last.book_close:%Y-%m-%d} ({paid}, FY {last.fiscal_year}): "
                         "the price is lowered by about the bonus on that day.")
        elif pd.notna(last.book_close) and last.book_close < day - pd.Timedelta(days=2 * 365):
            risks.append(f"No dividend recorded since FY {last.fiscal_year} ({paid}).")
        else:
            why.append(f"Dividends: last paid {paid} (FY {last.fiscal_year}).")
    upcoming = rights[(rights["symbol"] == pick.symbol) & (pd.to_datetime(rights["book_close"]) > day)]
    for bc in pd.to_datetime(upcoming["book_close"]):
        risks.append(f"Right-share book closure on {bc:%Y-%m-%d}: apply through MeroShare or sell before, or lose value.")
    if lockins is not None and not lockins.empty:
        ends = pd.to_datetime(lockins.loc[lockins["symbol"] == pick.symbol, "promoter_lock_end"]).dropna()
        for end in ends[(ends > day) & (ends <= day + pd.Timedelta(days=120))]:
            risks.append(f"Promoter shares unlock on {end:%Y-%m-%d}: extra supply can weigh on the price.")
    if pick.range52 > 0.9:
        risks.append("Near its 52-week high: less room before past selling levels.")
    return why, risks


def track_record(stocks, reports, calendar, market, start, end, hold=240, top=3, **screen):
    """Replay the screen quarterly from `start` to `end`; each pick held `hold` trading days vs `market`."""
    rows = []
    for first in pd.date_range(start, end, freq="QS"):
        days = calendar[calendar >= first]
        if len(days) <= hold:
            break
        day, exit_day = days[0], days[hold]
        picks = daily_picks(stocks, {}, reports, day, top=top, **screen)
        market_return = market.loc[exit_day] / market.loc[day] - 1
        for symbol in picks["symbol"]:
            close = stocks[symbol]["close"]
            rows.append({"start": day, "symbol": symbol, "return": close.loc[:exit_day].iloc[-1] / close.loc[day] - 1,
                         "market": market_return})
    return pd.DataFrame(rows, columns=["start", "symbol", "return", "market"])
