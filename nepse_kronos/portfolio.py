"""Daily rules engine: decide after each day's close, trade at the next market day's open."""
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from nepse_kronos.costs import CostModel

CIRCUIT_LIMIT = 0.095  # NEPSE daily limit is ±10%; opens beyond 9.5% are treated as unfillable
TRADE_COLUMNS = ["date", "symbol", "side", "shares", "price", "value", "fees", "tax", "reason"]
MIN_HISTORY_FOR_STATS = 20


@dataclass(frozen=True)
class Rules:
    top_k: int = 20                  # most shares held at once; buys come only from the top_k of the ranking
    keep_rank: int = 50              # keep a holding while it stays within this rank
    stop_loss: float = 0.15          # sell when the close is this far below the highest close since buying
    time_stop_days: int = 60         # sell when held this many days and still at or below the buy price
    cooldown_days: int = 20          # don't re-buy a share this many days after selling it
    max_turnover_share: float = 0.10  # a position may not exceed this share of the stock's median daily turnover
    vol_window: int = 60             # days used for volatility and turnover
    settlement_days: int = 2         # sale money is usable after this many trading days (T+2)
    size_by_volatility: bool = True  # put less money into jumpy shares
    tax_wait_days: int = 0           # delay a profitable ranking sale this close to the 1-year (lower tax) mark
    max_per_sector: int | None = None  # most holdings from one sector (needs `sectors`)
    dividend_delay_days: int = 20    # trading days between the ex-date and the cash arriving


def ranking(signal):
    """Symbols best-first; ties alphabetical."""
    signal = signal.dropna()
    return sorted(signal.index, key=lambda s: (-signal[s], s))


def _number(value):
    return None if pd.isna(value) else float(value)


def _waiting_for_lower_tax(held, price, day, rules, costs):
    """A profitable holding a few days short of the long-term tax rate is worth keeping a little longer."""
    if rules.tax_wait_days <= 0 or price * held["shares"] <= held["cost_basis"]:
        return False
    days_left = costs.long_term_days - (day - held["bought"]).days
    return 0 <= days_left <= rules.tax_wait_days


def _dividends_by_day(dividends, calendar):
    """{trading day: [(symbol, cash per unit)]}; an ex-date on a holiday counts on the next trading day."""
    events = defaultdict(list)
    for symbol, per_unit in (dividends or {}).items():
        for ex_date, amount in per_unit.items():
            pos = calendar.searchsorted(pd.Timestamp(ex_date))
            if pos < len(calendar) and amount > 0:
                events[calendar[pos]].append((symbol, float(amount)))
    return events


def simulate(panel, calendar, signals, rules=Rules(), capital=1_000_000.0, costs=None, return_state=False,
             dividends=None, sectors=None, blocklist=None, exposure=None, cash_rate=None):
    """Run the rules from the first signal date to the end of the calendar.

    Returns the daily portfolio value (cash + unsettled sale money + holdings at the close)
    and every trade with the reason for it. With return_state=True, also returns the position
    after the last close: holdings (shares), cash, unsettled sale money, and the orders queued
    for the next morning's open.

    Optional inputs (all off by default):
      dividends: {symbol: Series(ex_date -> cash per unit)} paid to holders at the previous close
      sectors:   {symbol: sector} used with Rules.max_per_sector
      blocklist: day -> set of symbols not to buy that day (quality, new supply, pumping filters)
      exposure:  day -> fraction (0-1] of the top_k slots to fill (money-cycle score)
      cash_rate: day -> annual interest rate earned on settled cash (after interest tax), compounded daily
    """
    costs = costs or CostModel()
    symbols = sorted(set().union(*(s.index for s in signals.values())))

    def matrix(column_of):
        return pd.DataFrame({s: column_of(panel[s]) for s in symbols}).reindex(calendar)

    closes = matrix(lambda df: df["close"]).ffill()
    prev_close = closes.shift(1)
    opens = matrix(lambda df: df["open"])
    window, min_rows = rules.vol_window, MIN_HISTORY_FOR_STATS
    vol = matrix(lambda df: df["close"].pct_change().rolling(window, min_periods=min_rows).std()).ffill()
    turnover = matrix(lambda df: df["amount"].rolling(window, min_periods=min_rows).median()).ffill()
    turnover_before = turnover.shift(1)
    max_rate = max(rate for _, rate in costs.commission_tiers)
    dividend_events = _dividends_by_day(dividends, calendar)
    sectors = sectors or {}

    def tradable(symbol, day, side):
        price = opens.at[day, symbol]
        if pd.isna(price):
            return False
        prev = prev_close.at[day, symbol]
        if pd.isna(prev):
            return True
        move = price / prev - 1.0
        return move < CIRCUIT_LIMIT if side == "buy" else move > -CIRCUIT_LIMIT

    cash, unsettled = float(capital), []        # unsettled: (day index when usable, amount)
    holdings, sell_queue, buy_queue = {}, {}, []
    cooldown_until, order, rank = {}, None, {}
    trades, equity = [], {}

    def settle(i):
        nonlocal cash, unsettled
        cash += sum(amount for when, amount in unsettled if when <= i)
        unsettled = [(when, amount) for when, amount in unsettled if when > i]

    previous_day = None
    for i, day in enumerate(calendar[calendar >= min(signals)]):
        # --- at the open: interest on idle cash, dividends to yesterday's holders, settle, sell, then buy ---
        if cash_rate is not None and previous_day is not None and cash > 0:
            daily = cash_rate(previous_day) * (1 - costs.interest_tax) / 365
            cash *= (1 + daily) ** (day - previous_day).days
        previous_day = day
        for symbol, per_unit in dividend_events.get(day, []):
            if symbol in holdings:
                units = holdings[symbol]["shares"]
                gross = units * per_unit
                tax = gross * costs.dividend_tax
                unsettled.append((i + rules.dividend_delay_days, gross - tax))
                trades.append([day, symbol, "dividend", units, per_unit, gross, 0.0, tax, "dividend"])
        settle(i)
        for symbol in sorted(sell_queue):
            if not tradable(symbol, day, "sell"):
                continue
            held, reason = holdings.pop(symbol), sell_queue.pop(symbol)
            price = opens.at[day, symbol]
            gross = held["shares"] * price
            fees = costs.trade_cost(gross, _number(turnover_before.at[day, symbol]))
            tax = costs.capital_gains_tax(gross - fees, held["cost_basis"], (day - held["bought"]).days)
            unsettled.append((i + rules.settlement_days, gross - fees - tax))
            cooldown_until[symbol] = i + rules.cooldown_days
            trades.append([day, symbol, "sell", held["shares"], price, gross, fees, tax, reason])
        settle(i)
        for symbol, budget in buy_queue:
            if symbol in holdings or not tradable(symbol, day, "buy"):
                continue
            price = opens.at[day, symbol]
            daily = _number(turnover_before.at[day, symbol])
            spend = min(budget, cash)
            unit = price * (1 + max_rate + costs.sebon_rate + costs.slippage_rate(spend, daily))
            shares = math.floor((spend - costs.dp_charge) / unit)
            if shares <= 0:
                continue
            gross = shares * price
            fees = costs.trade_cost(gross, daily)
            cash -= gross + fees
            holdings[symbol] = {"shares": shares, "cost_basis": gross + fees, "entry": price, "high": price,
                                "days": 0, "bought": day, "rank_exit": False}
            trades.append([day, symbol, "buy", shares, price, gross, fees, 0.0, "buy"])
        buy_queue = []

        # --- at the close: value the portfolio, then decide tomorrow's trades ---
        close = closes.loc[day]
        for symbol, held in holdings.items():
            held["high"] = max(held["high"], close[symbol])
            held["days"] += 1
        value = cash + sum(amount for _, amount in unsettled) + sum(
            held["shares"] * close[symbol] for symbol, held in holdings.items())
        equity[day] = value

        refreshed = day in signals
        if refreshed:
            order = ranking(signals[day])
            rank = {symbol: k + 1 for k, symbol in enumerate(order)}
        for symbol, held in holdings.items():
            if symbol in sell_queue:
                continue
            if close[symbol] <= held["high"] * (1 - rules.stop_loss):
                sell_queue[symbol] = "stop_loss"
                continue
            if held["days"] >= rules.time_stop_days and close[symbol] <= held["entry"]:
                sell_queue[symbol] = "time_limit"
                continue
            if refreshed:
                held["rank_exit"] = rank.get(symbol, math.inf) > rules.keep_rank
            if held["rank_exit"] and not _waiting_for_lower_tax(held, close[symbol], day, rules, costs):
                sell_queue[symbol] = "rank"

        if order is None:
            continue
        target = rules.top_k if exposure is None else round(rules.top_k * min(1.0, max(0.0, exposure(day))))
        slots = target - (len(holdings) - len(sell_queue))
        blocked = blocklist(day) if blocklist else set()
        per_sector = Counter(sectors.get(s) for s in holdings if s not in sell_queue)
        ranked_vol = vol.loc[day, order]
        typical_vol = ranked_vol.median() if ranked_vol.notna().any() else math.nan
        considered = 0  # buys come from the best top_k names that pass the filters
        for symbol in order:
            if slots <= 0 or considered >= rules.top_k:
                break
            sector = sectors.get(symbol)
            sector_full = (rules.max_per_sector and sector is not None
                           and per_sector[sector] >= rules.max_per_sector and symbol not in holdings)
            if symbol in blocked or sector_full:
                continue
            considered += 1
            if symbol in holdings or cooldown_until.get(symbol, -1) > i or pd.isna(close[symbol]):
                continue
            budget = value / rules.top_k
            own_vol = vol.at[day, symbol]
            if rules.size_by_volatility and own_vol > 0 and typical_vol > 0:
                budget *= min(1.0, typical_vol / own_vol)
            daily = turnover.at[day, symbol]
            if daily > 0:
                budget = min(budget, rules.max_turnover_share * daily)
            buy_queue.append((symbol, budget))
            per_sector[sector] += 1
            slots -= 1

    result = (pd.Series(equity, name="equity").rename_axis("timestamps"),
              pd.DataFrame(trades, columns=TRADE_COLUMNS))
    if not return_state:
        return result
    state = {
        "holdings": {symbol: held["shares"] for symbol, held in holdings.items()},
        "cash": cash,
        "unsettled": sum(amount for _, amount in unsettled),
        "sell_orders": dict(sell_queue),
        "buy_orders": list(buy_queue),
    }
    return (*result, state)
