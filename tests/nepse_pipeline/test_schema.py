import pandas as pd
import pytest

from nepse_kronos.schema import CANONICAL_COLUMNS, normalize_ohlcv, trim_close_only_history


def test_maps_site_specific_column_names_and_parses_commas():
    raw = pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-01"],
        "Open": ["1,000.0", "990"],
        "High": ["1,010", "1,000"],
        "Low": ["995", "985"],
        "LTP": ["1,005", "995"],
        "Total Traded Quantity": ["2,000", "1,500"],
        "Turnover": ["2,010,000", "1,492,500"],
    })
    df = normalize_ohlcv(raw)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["timestamps"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert df.loc[1, "open"] == 1000.0
    assert df.loc[1, "close"] == 1005.0
    assert df.loc[1, "volume"] == 2000.0
    assert df.loc[1, "amount"] == 2010000.0


def test_handles_bom_in_first_header():
    raw = pd.DataFrame({"﻿timestamps": ["2024-01-01"], "open": [1], "high": [2], "low": [1], "close": [2]})
    assert normalize_ohlcv(raw)["timestamps"].iloc[0] == pd.Timestamp("2024-01-01")


def test_missing_volume_and_amount_are_filled():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [10], "high": [12], "low": [9], "close": [11]})
    df = normalize_ohlcv(raw)
    assert df.loc[0, "volume"] == 0.0
    assert df.loc[0, "amount"] == 0.0


def test_missing_price_column_raises():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [10], "high": [12], "low": [9]})
    with pytest.raises(ValueError, match="close"):
        normalize_ohlcv(raw)


def test_drops_bad_rows_and_duplicate_dates():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03"],
        "open": [10, "-", 0, 12],
        "high": [11, 11, 11, 13],
        "low": [9, 9, 9, 11],
        "close": [10, 10, 10, 12.5],
    })
    df = normalize_ohlcv(raw)
    # row 2 has a non-numeric open, row 3 has a zero price, 2024-01-03 keeps the last occurrence
    assert df["timestamps"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-03"]
    assert df.loc[1, "close"] == 12.5


def test_repairs_high_low_that_do_not_contain_open_close():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [10], "high": [10.5], "low": [9.8], "close": [11]})
    df = normalize_ohlcv(raw)
    assert df.loc[0, "high"] == 11.0
    assert df.loc[0, "low"] == 9.8


def test_trim_close_only_history_drops_leading_flat_candles_only():
    df = normalize_ohlcv(pd.DataFrame({
        "date": ["2016-11-24", "2016-11-25", "2016-11-28", "2016-11-29", "2016-11-30"],
        "open": [10, 11, 12, 13, 14],
        "high": [10, 11, 12.5, 13, 14.5],
        "low": [10, 11, 11.8, 13, 13.9],
        "close": [10, 11, 12.2, 13, 14.1],
    }))
    out = trim_close_only_history(df)
    # the two leading close-only rows go; the later flat row (2016-11-29) is a real thin-trading day and stays
    assert out["timestamps"].dt.strftime("%Y-%m-%d").tolist() == ["2016-11-28", "2016-11-29", "2016-11-30"]


def test_trim_close_only_history_keeps_series_without_real_candles():
    df = normalize_ohlcv(pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "open": [1, 2],
                                       "high": [1, 2], "low": [1, 2], "close": [1, 2]}))
    assert len(trim_close_only_history(df)) == 2
