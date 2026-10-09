"""A fair yardstick: the NEPSE index is a price index (no dividends), while our share prices include
cash dividends. This adds an estimate of the dividends an index holder would have received."""
import pandas as pd

from nepse_kronos.metrics import TRADING_DAYS_PER_YEAR


def market_dividend_yield(events, official_close, top_n=150, exclude=()):
    """Per calendar year: average cash-dividend yield across the `top_n` most-traded shares (non-payers count as 0).

    events: symbol, book_close, cash (Rs per share). official_close: {symbol: unadjusted close series}.
    "Most traded" = most trading days in that year. `exclude`: symbols that aren't ordinary shares
    (mutual funds have a Rs 10 face value, so their dividend % means something else).
    """
    exclude = set(exclude)
    official_close = {s: c for s, c in official_close.items() if s not in exclude}
    events = events.assign(year=pd.to_datetime(events["book_close"]).dt.year)
    cash = events.groupby(["symbol", "year"])["cash"].sum()
    yields = {}
    years = sorted({d.year for s in official_close.values() for d in s.index})
    for year in years:
        days_traded = {s: (c.index.year == year).sum() for s, c in official_close.items()}
        top = sorted((s for s, n in days_traded.items() if n > 0), key=lambda s: -days_traded[s])[:top_n]
        values = []
        for symbol in top:
            prices = official_close[symbol]
            price = prices[prices.index.year == year].median()
            values.append(cash.get((symbol, year), 0.0) / price if price > 0 else 0.0)
        yields[year] = sum(values) / len(values) if values else 0.0
    return pd.Series(yields, name="dividend_yield")


def total_return_index(index_close, yearly_yield):
    """Index level plus dividends, spreading each year's yield evenly over its trading days."""
    per_day = index_close.index.year.map(lambda y: yearly_yield.get(y, 0.0) / TRADING_DAYS_PER_YEAR)
    growth = pd.Series(1 + per_day.to_numpy(), index=index_close.index).shift(1).fillna(1.0).cumprod()
    return index_close * growth
