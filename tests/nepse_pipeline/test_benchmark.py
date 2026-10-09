import pandas as pd
import pytest

from nepse_kronos.benchmark import market_dividend_yield, total_return_index


def test_market_dividend_yield_averages_payers_and_non_payers():
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    official = {"PAY": pd.Series(200.0, index=days), "NONE": pd.Series(50.0, index=days),
                "TINY": pd.Series(100.0, index=days[:10])}            # rarely traded: outside the top 2
    events = pd.DataFrame({"symbol": ["PAY", "PAY"], "book_close": pd.to_datetime(["2024-03-01", "2024-09-01"]),
                           "cash": [4.0, 6.0]})
    yields = market_dividend_yield(events, official, top_n=2)
    assert yields.loc[2024] == pytest.approx((10.0 / 200.0 + 0.0) / 2)


def test_total_return_index_adds_the_yield_over_the_year():
    days = pd.bdate_range("2024-01-01", periods=240)
    index = pd.Series(1000.0, index=days)
    out = total_return_index(index, pd.Series({2024: 0.024}))
    assert out.iloc[0] == pytest.approx(1000.0)
    assert out.iloc[-1] == pytest.approx(1000.0 * (1 + 0.024 / 240) ** 239)


def test_market_dividend_yield_can_exclude_non_shares():
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    official = {"PAY": pd.Series(200.0, index=days), "FUND": pd.Series(10.0, index=days)}
    events = pd.DataFrame({"symbol": ["PAY", "FUND"], "book_close": pd.to_datetime(["2024-03-01", "2024-08-01"]),
                           "cash": [10.0, 50.0]})
    yields = market_dividend_yield(events, official, top_n=10, exclude={"FUND"})
    assert yields.loc[2024] == pytest.approx(0.05)
