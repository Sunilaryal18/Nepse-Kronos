import numpy as np
import pandas as pd
import pytest

from nepse_kronos.costs import CostModel
from nepse_kronos.my_portfolio import health_notes, load_holdings, summarize, value_holdings


def _stocks(ohlcv_factory, n=120):
    return {"UP": ohlcv_factory(n, close=np.linspace(200, 260, n)).set_index("timestamps"),
            "DOWN": ohlcv_factory(n, close=np.linspace(300, 240, n)).set_index("timestamps")}


def test_load_holdings(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("symbol,units,buy_price,note\nup,100,250,\nDOWN,50,290,from memory\n")
    h = load_holdings(path)
    assert h["symbol"].tolist() == ["UP", "DOWN"] and h["units"].tolist() == [100, 50]


def test_value_holdings_gain_and_todays_move(ohlcv_factory):
    stocks = _stocks(ohlcv_factory)
    day = stocks["UP"].index[-1]
    holdings = pd.DataFrame({"symbol": ["UP", "DOWN", "MISSING"], "units": [100, 50, 10],
                             "buy_price": [250.0, 290.0, 100.0], "note": ["", "", ""]})
    v = value_holdings(holdings, stocks, day, {"UP": "Hydropower"}).set_index("symbol")
    up = stocks["UP"]["close"]
    assert v.loc["UP", "value"] == pytest.approx(100 * up.iloc[-1])
    assert v.loc["UP", "gain"] == pytest.approx(100 * (up.iloc[-1] - 250))
    assert v.loc["UP", "day_change"] == pytest.approx(up.iloc[-1] / up.iloc[-2] - 1)
    assert v.loc["DOWN", "gain_pct"] == pytest.approx(stocks["DOWN"]["close"].iloc[-1] / 290 - 1)
    assert pd.isna(v.loc["MISSING", "last"])                                  # unknown symbol: kept, not valued


def test_summary_and_take_home_after_fees_and_tax(ohlcv_factory):
    stocks = _stocks(ohlcv_factory)
    day = stocks["UP"].index[-1]
    holdings = pd.DataFrame({"symbol": ["UP", "DOWN"], "units": [100, 50], "buy_price": [250.0, 290.0], "note": ""})
    v = value_holdings(holdings, stocks, day, {})
    s = summarize(v, CostModel())
    assert s["cost"] == pytest.approx(100 * 250 + 50 * 290)
    assert s["value"] == pytest.approx(v["value"].sum())
    fees = sum(CostModel().trade_cost(x) for x in v["value"])
    up_gain_after_fees = v.loc[0, "value"] - CostModel().trade_cost(v.loc[0, "value"]) - 100 * 250
    assert s["take_home"] == pytest.approx(s["value"] - fees - 0.10 * max(up_gain_after_fees, 0))


def test_health_notes_flag_trend_and_valuation(ohlcv_factory):
    stocks = _stocks(ohlcv_factory)
    day = stocks["UP"].index[-1]
    reports = pd.DataFrame({"symbol": ["DOWN", "DOWN"], "fiscal_year": ["a", "b"], "quarter": [1, 1],
                            "eps": [5.0, 4.0], "net_worth_per_share": [100.0, 100.0],
                            "submitted": [day - pd.Timedelta(days=400), day - pd.Timedelta(days=30)]})
    down = " ".join(health_notes("DOWN", stocks, reports, day))
    assert "falling" in down and "60× profit" in down and "profit -20%" in down
    assert "No published profit report" in " ".join(health_notes("UP", stocks, reports, day))
