from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nepse_kronos.costs import CostModel
from nepse_kronos.portfolio import Rules, simulate

FREE = CostModel(commission_tiers=((float("inf"), 0.0),), sebon_rate=0.0, dp_charge=0.0, slippage=0.0,
                 impact=0.0, cgt_rate=0.0)
# One share at a time, no extra rules unless a test turns them on
BASE = Rules(top_k=1, keep_rank=1, stop_loss=0.15, time_stop_days=1000, cooldown_days=0,
             max_turnover_share=1.0, settlement_days=0, size_by_volatility=False)


def _panel(ohlcv_factory, **closes):
    # open == close in the factory, so each day's open equals that day's close
    return {sym: ohlcv_factory(len(c), start="2024-01-01", close=c).set_index("timestamps") for sym, c in closes.items()}


def _sig(*order):
    """Signal ranking the given symbols best-first."""
    return pd.Series({sym: float(len(order) - k) for k, sym in enumerate(order)})


def _rows(trades, cols=("date", "symbol", "side", "reason")):
    return trades[list(cols)].values.tolist()


def test_buys_top_pick_at_next_open(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100, 105, 110.25, 110.25, 110.25], B=[100] * 5, C=[100, 95, 90.25, 90.25, 90.25])
    cal = panel["B"].index
    equity, trades = simulate(panel, cal, {cal[0]: _sig("A", "B", "C")}, BASE, capital=10_000, costs=FREE)

    assert trades[["date", "symbol", "side", "shares", "price"]].values.tolist() == [[cal[1], "A", "buy", 95, 105.0]]
    assert equity[cal[0]] == 10_000
    assert equity[cal[1]] == pytest.approx(25 + 95 * 105)
    assert equity[cal[2]] == pytest.approx(25 + 95 * 110.25)


def test_keep_zone_holds_until_rank_drops_below_keep_rank(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 7, B=[100] * 7, C=[100] * 7, D=[100] * 7)
    cal = panel["A"].index
    signals = {cal[0]: _sig("A", "B", "C", "D"), cal[2]: _sig("B", "C", "A", "D"), cal[4]: _sig("B", "C", "D", "A")}
    _, trades = simulate(panel, cal, signals, replace(BASE, keep_rank=3), capital=10_000, costs=FREE)

    assert _rows(trades) == [[cal[1], "A", "buy", "buy"], [cal[5], "A", "sell", "rank"], [cal[5], "B", "buy", "buy"]]


def test_loss_limit_from_highest_close(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100, 100, 109, 103, 92, 92, 92])
    cal = panel["A"].index
    _, trades = simulate(panel, cal, {cal[0]: _sig("A")}, replace(BASE, cooldown_days=10), capital=10_000, costs=FREE)

    # highest close 109 -> limit 92.65; the close of 92 on day 4 triggers a sale at day 5's open
    assert _rows(trades) == [[cal[1], "A", "buy", "buy"], [cal[5], "A", "sell", "stop_loss"]]


def test_time_limit_and_cool_off(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 9)
    cal = panel["A"].index
    _, trades = simulate(panel, cal, {cal[0]: _sig("A")}, replace(BASE, time_stop_days=3, cooldown_days=2),
                         capital=10_000, costs=FREE)

    # held 3 closes (days 1-3) without gain -> sold day 4; cool-off until day 6's close -> re-bought day 7
    assert _rows(trades) == [[cal[1], "A", "buy", "buy"], [cal[4], "A", "sell", "time_limit"],
                             [cal[7], "A", "buy", "buy"]]


def test_sale_money_settles_after_two_days(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 7, B=[100] * 7)
    cal = panel["A"].index
    equity, trades = simulate(panel, cal, {cal[0]: _sig("A", "B"), cal[2]: _sig("B", "A")},
                              replace(BASE, settlement_days=2), capital=10_000, costs=FREE)

    assert _rows(trades) == [[cal[1], "A", "buy", "buy"], [cal[3], "A", "sell", "rank"], [cal[5], "B", "buy", "buy"]]
    assert equity[cal[4]] == pytest.approx(10_000)  # unsettled money still counts as ours


def test_upper_price_limit_delays_buy(ohlcv_factory):
    panel = _panel(ohlcv_factory, B=[100, 110, 110, 110, 110])
    cal = panel["B"].index
    _, trades = simulate(panel, cal, {cal[0]: _sig("B")}, BASE, capital=10_000, costs=FREE)
    assert _rows(trades) == [[cal[2], "B", "buy", "buy"]]


def test_sale_retried_when_share_does_not_trade(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 6, B=[100] * 6)
    cal = panel["B"].index
    panel["A"] = panel["A"].drop(cal[3])
    _, trades = simulate(panel, cal, {cal[0]: _sig("A", "B"), cal[2]: _sig("B", "A")}, BASE,
                         capital=10_000, costs=FREE)
    assert _rows(trades) == [[cal[1], "A", "buy", "buy"], [cal[4], "A", "sell", "rank"], [cal[4], "B", "buy", "buy"]]


def test_jumpy_shares_get_less_money_and_turnover_caps_size(ohlcv_factory):
    n = 40
    panel = _panel(ohlcv_factory, CALM=np.linspace(100, 101, n), JUMPY=100 * (1 + 0.03 * (-1) ** np.arange(n)))
    cal = panel["CALM"].index
    rules = replace(BASE, top_k=2, keep_rank=2, size_by_volatility=True)
    _, trades = simulate(panel, cal, {cal[30]: _sig("CALM", "JUMPY")}, rules, capital=10_000, costs=FREE)
    value = trades.set_index("symbol")["value"]
    assert value["JUMPY"] < 0.5 * value["CALM"]

    # daily turnover is close * 1000 (~100,000); 1% of it caps each buy at ~1,000
    _, capped = simulate(panel, cal, {cal[30]: _sig("CALM", "JUMPY")}, replace(rules, max_turnover_share=0.01),
                         capital=10_000, costs=FREE)
    assert (capped["value"] <= 1_010).all()


def test_fees_and_tax_follow_cost_model(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100, 100, 200, 200, 200], B=[100] * 5)
    cal = panel["B"].index
    costs = CostModel(impact=0.0)
    _, trades = simulate(panel, cal, {cal[0]: _sig("A", "B"), cal[2]: _sig("B", "A")}, BASE,
                         capital=100_000, costs=costs)

    buy, sell = trades.iloc[0], trades.iloc[1]
    assert buy["shares"] == 993
    assert buy["fees"] == pytest.approx(costs.trade_cost(993 * 100))
    gross = 993 * 200
    fees = costs.trade_cost(gross)
    assert sell["fees"] == pytest.approx(fees)
    assert sell["tax"] == pytest.approx(0.10 * (gross - fees - (993 * 100 + buy["fees"])))


def test_return_state_reports_holdings_and_next_morning_orders(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 5, B=[100] * 5)
    cal = panel["A"].index
    # A bought on day 1; ranking on the last day prefers B -> sell A and buy B tomorrow
    _, _, state = simulate(panel, cal, {cal[0]: _sig("A", "B"), cal[4]: _sig("B", "A")}, BASE,
                           capital=10_000, costs=FREE, return_state=True)
    assert state["holdings"] == {"A": 100}
    assert state["sell_orders"] == {"A": "rank"}
    assert [symbol for symbol, _ in state["buy_orders"]] == ["B"]
    assert state["cash"] == pytest.approx(0.0)


def _long_panel(ohlcv_factory, n=320):
    # A doubles early and then stays flat; B stays flat. ~320 weekdays is about 15 months.
    a = np.r_[np.full(2, 100.0), np.linspace(100, 200, 50), np.full(n - 52, 200.0)]
    return _panel(ohlcv_factory, A=a, B=np.full(n, 100.0))


def test_long_term_tax_rate_after_one_year(ohlcv_factory):
    panel = _long_panel(ohlcv_factory)
    cal = panel["A"].index
    costs = replace(FREE, cgt_rate=0.10, cgt_rate_long=0.075)
    _, trades = simulate(panel, cal, {cal[0]: _sig("A", "B"), cal[300]: _sig("B", "A")}, BASE,
                         capital=10_000, costs=costs)
    sell = trades[trades["side"] == "sell"].iloc[0]
    assert (sell["date"] - cal[1]).days > 365
    assert sell["tax"] == pytest.approx(0.075 * (100 * 200 - 100 * 100))


def test_tax_wait_delays_profitable_rank_sale_until_one_year(ohlcv_factory):
    panel = _long_panel(ohlcv_factory)
    cal = panel["A"].index
    signals = {cal[0]: _sig("A", "B"), cal[230]: _sig("B", "A")}   # ranking drops A ~10.5 months after buying
    costs = replace(FREE, cgt_rate=0.10, cgt_rate_long=0.075)

    _, now = simulate(panel, cal, signals, BASE, capital=10_000, costs=costs)
    assert now[now["side"] == "sell"]["date"].tolist() == [cal[231]]

    _, waited = simulate(panel, cal, signals, replace(BASE, tax_wait_days=60), capital=10_000, costs=costs)
    sell = waited[waited["side"] == "sell"].iloc[0]
    assert sell["reason"] == "rank"
    assert 365 < (sell["date"] - cal[1]).days <= 370
    assert sell["tax"] == pytest.approx(0.075 * (100 * 200 - 100 * 100))


def test_cash_dividend_paid_to_holders_after_tax_and_delay(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 30)
    cal = panel["A"].index
    dividends = {"A": pd.Series({cal[5]: 10.0})}          # Rs 10 per unit, ex-date day 5
    costs = replace(FREE, dividend_tax=0.05)
    rules = replace(BASE, dividend_delay_days=20)
    equity, trades = simulate(panel, cal, {cal[0]: _sig("A")}, rules, capital=10_000, costs=costs,
                              dividends=dividends)
    div = trades[trades["side"] == "dividend"]
    assert div[["date", "symbol", "shares"]].values.tolist() == [[cal[5], "A", 100]]
    assert div["value"].iloc[0] == pytest.approx(1_000.0)
    assert div["tax"].iloc[0] == pytest.approx(50.0)
    assert equity[cal[5]] == pytest.approx(10_000 + 950)        # counted as ours while waiting to be paid


def test_no_dividend_if_sold_before_ex_date(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 10, B=[100] * 10)
    cal = panel["A"].index
    _, trades = simulate(panel, cal, {cal[0]: _sig("A", "B"), cal[2]: _sig("B", "A")}, BASE, capital=10_000,
                         costs=FREE, dividends={"A": pd.Series({cal[5]: 10.0})})
    assert "dividend" not in trades["side"].tolist()


def test_sector_limit(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 5, B=[100] * 5, C=[100] * 5)
    cal = panel["A"].index
    rules = replace(BASE, top_k=2, keep_rank=3, max_per_sector=1)
    _, trades = simulate(panel, cal, {cal[0]: _sig("A", "B", "C")}, rules, capital=10_000, costs=FREE,
                         sectors={"A": "Bank", "B": "Bank", "C": "Hydro"})
    assert sorted(trades["symbol"]) == ["A", "C"]        # B skipped: one bank already


def test_blocklist_prevents_buying(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 5, B=[100] * 5)
    cal = panel["A"].index
    _, trades = simulate(panel, cal, {cal[0]: _sig("A", "B")}, BASE, capital=10_000, costs=FREE,
                         blocklist=lambda day: {"A"})
    assert trades["symbol"].tolist() == ["B"]


def test_exposure_limits_number_of_new_positions(ohlcv_factory):
    panel = _panel(ohlcv_factory, **{s: [100] * 5 for s in "ABCD"})
    cal = panel["A"].index
    rules = replace(BASE, top_k=4, keep_rank=4)
    _, trades = simulate(panel, cal, {cal[0]: _sig("A", "B", "C", "D")}, rules, capital=10_000, costs=FREE,
                         exposure=lambda day: 0.5)
    assert sorted(trades["symbol"]) == ["A", "B"]        # 50% of 4 slots


def test_idle_cash_earns_interest_after_tax(ohlcv_factory):
    panel = _panel(ohlcv_factory, A=[100] * 11)
    cal = panel["A"].index                                      # Mon 2024-01-01 .. Mon 2024-01-15
    costs = replace(FREE, interest_tax=0.06)
    equity, trades = simulate(panel, cal, {cal[0]: _sig("A")}, BASE, capital=10_000, costs=costs,
                              exposure=lambda day: 0.0, cash_rate=lambda day: 0.10)
    assert trades.empty
    days = (cal[-1] - cal[0]).days                             # 14 calendar days, interest also over weekends
    assert equity[cal[-1]] == pytest.approx(10_000 * (1 + 0.10 * 0.94 / 365) ** days, rel=1e-9)
