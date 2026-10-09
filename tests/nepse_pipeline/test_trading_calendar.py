import pandas as pd

from nepse_kronos.trading_calendar import load_holidays, next_trading_days


def _dates(series):
    return series.dt.strftime("%Y-%m-%d").tolist()


def test_skips_weekend_after_friday():
    # 2025-01-03 is a Friday
    assert _dates(next_trading_days("2025-01-03", 3)) == ["2025-01-06", "2025-01-07", "2025-01-08"]


def test_skips_holidays():
    out = next_trading_days("2025-01-03", 3, holidays=[pd.Timestamp("2025-01-07")])
    assert _dates(out) == ["2025-01-06", "2025-01-08", "2025-01-09"]


def test_result_is_named_series_without_time_of_day():
    out = next_trading_days(pd.Timestamp("2025-01-06 15:00"), 2)
    assert out.name == "timestamps"
    assert _dates(out) == ["2025-01-07", "2025-01-08"]
    assert (out.dt.hour == 0).all()


def test_load_holidays_reads_csv_and_ignores_comments(tmp_path):
    path = tmp_path / "holidays.csv"
    path.write_text("# comment line\ndate,name\n2025-01-07,Test Day\n2025-01-01,New Year\n")
    assert load_holidays(path) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-07")]


def test_load_holidays_missing_file_returns_empty(tmp_path):
    assert load_holidays(tmp_path / "nope.csv") == []
