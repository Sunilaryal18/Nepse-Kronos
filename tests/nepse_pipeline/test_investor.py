import numpy as np
import pandas as pd
import pytest

from nepse_kronos.investor import (broker_tiebreak, money_exposure, pump_blocklist, quality_blocklist,
                                   supply_blocklist)

D = pd.Timestamp


def _prices(ohlcv_factory, **closes):
    return {s: ohlcv_factory(len(c), start="2024-01-01", close=c).set_index("timestamps") for s, c in closes.items()}


def test_quality_uses_only_published_reports(ohlcv_factory):
    panel = _prices(ohlcv_factory, GOOD=[100] * 40, LOSS=[100] * 40, PRICEY=[100] * 40, NEW=[100] * 40)
    days = panel["GOOD"].index
    reports = pd.DataFrame({
        "symbol": ["GOOD", "LOSS", "LOSS", "PRICEY", "PRICEY"],
        "eps": [10.0, -2.0, 5.0, 1.0, 10.0],
        "net_worth_per_share": [100.0, 100.0, 100.0, 100.0, 10.0],
        "submitted": [days[5], days[5], days[20], days[5], days[20]],
    })
    blocked = quality_blocklist(reports, panel)
    assert blocked(days[4]) == set()                       # nothing published yet
    assert blocked(days[6]) == {"LOSS", "PRICEY"}          # loss; P/E 100 > 40
    assert blocked(days[21]) == {"PRICEY"}                  # LOSS now profitable; PRICEY price/book 10 > 5
    assert "NEW" not in blocked(days[30])                   # no report -> not judged


def test_supply_blocks_before_lockin_end_and_right_book_close():
    lockins = pd.DataFrame({"symbol": ["A"], "promoter_lock_end": [D("2025-03-01")]})
    rights = pd.DataFrame({"symbol": ["B"], "book_close": [D("2025-03-01")]})
    blocked = supply_blocklist(lockins, rights)
    assert blocked(D("2024-12-15")) == set()
    assert blocked(D("2025-01-15")) == {"A"}               # 45 days before promoter unlock
    assert blocked(D("2025-02-10")) == {"A", "B"}          # 19 days before right book close
    assert blocked(D("2025-03-02")) == set()


def test_pump_blocklist_needs_concentration_and_price_jump(ohlcv_factory):
    n = 30
    panel = _prices(ohlcv_factory, PUMP=np.linspace(100, 140, n), CALM=np.full(n, 100.0),
                    RISER=np.linspace(100, 140, n), **{f"X{k}": np.full(n, 100.0) for k in range(20)})
    day = panel["PUMP"].index[-1]
    rows = [{"date": day, "symbol": s, "buyer_hhi": 0.05, "top5_net_buy": 0.0} for s in panel]
    feats = pd.DataFrame(rows)
    feats.loc[feats.symbol == "PUMP", "buyer_hhi"] = 0.9
    feats.loc[feats.symbol == "CALM", "buyer_hhi"] = 0.9
    assert pump_blocklist(feats, panel)(day) == {"PUMP"}   # CALM concentrated but flat; RISER rose but spread out


def test_money_exposure_scores_published_conditions_only():
    months = pd.date_range("2023-01-16", periods=16, freq="MS") + pd.Timedelta(days=15)
    macro = pd.DataFrame({
        "month": months,
        "interbank": [5.0] * 15 + [1.0],      # last month: cheaper money than the 12-month average
        "cd_ratio": [80.0] * 15 + [70.0],     # more room to lend
        "lending_rate": [10.0] * 15 + [9.0],  # falling vs 3 months earlier
    })
    macro["available_from"] = macro["month"] + pd.Timedelta(days=40)
    exposure = money_exposure(macro)
    last = macro["available_from"].iloc[-1]
    assert exposure(D("2022-06-01")) == 1.0                         # before any data
    assert exposure(last - pd.Timedelta(days=1)) == pytest.approx(0.5)   # last month not yet published: score 0
    assert exposure(last) == pytest.approx(1.0)                      # score 3 -> fully invested


def test_broker_tiebreak_reorders_only_close_ranks():
    fillers = {f"F{k}": 0.2 - k * 0.001 for k in range(38)}
    signal = pd.Series({"A": 0.30, "B": 0.29, "C": 0.10, **fillers}, name="pred_return")
    feats = pd.DataFrame({"date": [D("2025-01-01")] * 41, "symbol": ["A", "B", "C", *fillers],
                          "top5_net_buy": [0.2, 0.6, 0.6, *([0.2] * 38)]})  # median 0.2: A neutral, B and C heavy
    order = list(broker_tiebreak(signal, feats, D("2025-01-02")).sort_values(ascending=False).index)
    assert order[:2] == ["B", "A"]            # neighbours swap on broker buying
    assert order.index("C") > 30              # a weak share is not lifted far


def test_broker_tiebreak_compares_with_the_typical_share_that_day():
    # top5_net_buy is never negative; "strong buying" means above the day's median
    fillers = {f"F{k}": 0.2 - k * 0.001 for k in range(38)}
    signal = pd.Series({"A": 0.30, "B": 0.29, **fillers}, name="pred_return")
    feats = pd.DataFrame({"date": [D("2025-01-01")] * 40, "symbol": ["A", "B", *fillers],
                          "top5_net_buy": [0.05, 0.60, *([0.20] * 38)]})
    order = list(broker_tiebreak(signal, feats, D("2025-01-02")).sort_values(ascending=False).index)
    assert order[:2] == ["B", "A"] or order[0] == "B"
    assert order.index("A") > order.index("B")


def test_deposit_rate_uses_published_data_only():
    from nepse_kronos.investor import deposit_rate
    macro = pd.DataFrame({"deposit_rate": [5.0, 8.0], "available_from": [D("2024-02-10"), D("2024-03-10")]})
    rate = deposit_rate(macro)
    assert rate(D("2024-01-01")) == 0.0
    assert rate(D("2024-02-20")) == pytest.approx(0.05)
    assert rate(D("2024-03-10")) == pytest.approx(0.08)


def test_money_exposure_average_window_is_adjustable():
    months = pd.date_range("2023-01-16", periods=8, freq="MS") + pd.Timedelta(days=15)
    macro = pd.DataFrame({"month": months, "interbank": [9.0] * 6 + [2.0, 3.0], "cd_ratio": 80.0, "lending_rate": 10.0})
    macro["available_from"] = macro["month"] + pd.Timedelta(days=40)
    last = macro["available_from"].iloc[-1]
    assert money_exposure(macro, months=12)(last) == pytest.approx(0.5 + 0.5 / 3)  # 3.0 < avg of 9s and 2.0
    assert money_exposure(macro, months=1)(last) == pytest.approx(0.5)            # 3.0 is not below last month's 2.0
