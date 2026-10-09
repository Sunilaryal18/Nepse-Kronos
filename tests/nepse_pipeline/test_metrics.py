import numpy as np
import pandas as pd
import pytest

from nepse_kronos.metrics import TRADING_DAYS_PER_YEAR, performance, rank_ic, summarize_ic, trade_summary, yearly_table


def test_performance_returns_and_drawdown():
    idx = pd.bdate_range("2024-01-01", periods=4)
    equity = pd.Series([100.0, 120.0, 90.0, 135.0], index=idx)
    flat = pd.Series(100.0, index=idx)
    p = performance(equity, flat)
    assert p["strategy"]["total_return"] == pytest.approx(0.35)
    assert p["strategy"]["max_drawdown"] == pytest.approx(-0.25)
    assert p["benchmark"]["total_return"] == 0.0
    assert np.isnan(p["benchmark"]["sharpe"])


def test_cagr_over_one_trading_year():
    idx = pd.bdate_range("2024-01-01", periods=TRADING_DAYS_PER_YEAR + 1)
    equity = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    assert performance(equity, equity)["strategy"]["cagr"] == pytest.approx(1.0)


def test_rank_ic_perfect_and_reversed(ohlcv_factory):
    n = 30
    panel = {f"S{k}": ohlcv_factory(n, close=np.linspace(100, 100 * (1 + g), n)).set_index("timestamps")
             for k, g in enumerate([0.0, 0.1, 0.2, 0.3, 0.4])}
    cal = panel["S0"].index
    good = pd.Series({f"S{k}": float(k) for k in range(5)})
    ic = rank_ic({cal[10]: good, cal[12]: -good}, panel, cal, horizon=5)
    assert ic[cal[10]] == pytest.approx(1.0)
    assert ic[cal[12]] == pytest.approx(-1.0)
    assert summarize_ic(ic)["periods"] == 2


def test_trade_summary():
    idx = pd.bdate_range("2024-01-01", periods=TRADING_DAYS_PER_YEAR + 1)
    equity = pd.Series(1000.0, index=idx)
    trades = pd.DataFrame({"side": ["buy", "sell", "buy", "sell"], "value": [500.0, 500.0, 500.0, 400.0],
                           "fees": [1.0, 1.0, 1.0, 1.0], "tax": [0.0, 2.0, 0.0, 0.0],
                           "reason": ["buy", "stop_loss", "buy", "rank"]})
    assert trade_summary(trades, equity) == {"trades": 4, "fees": 4.0, "tax": 2.0, "dividends": 0.0, "annual_turnover": 1.0,
                                             "sold_rank": 1, "sold_stop_loss": 1, "sold_time_limit": 0}


def test_yearly_table():
    idx = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-06-30"])
    equity = pd.Series([100.0, 110.0, 121.0, 99.0], index=idx)
    index = pd.Series([50.0, 50.0, 55.0, 60.0], index=idx)
    table = yearly_table(equity, index)
    assert list(table.index) == [2024, 2025]
    assert table.loc[2024, "strategy"] == pytest.approx(0.10)
    assert table.loc[2025, "strategy"] == pytest.approx(99 / 110 - 1)
    assert table.loc[2025, "benchmark"] == pytest.approx(0.20)
