import pandas as pd

from nepse_kronos.universe import liquid_universe, load_panel, rebalance_dates


def test_load_panel_indexes_by_date_and_skips_indices(tmp_path, ohlcv_factory):
    ohlcv_factory(5).to_csv(tmp_path / "ABC.csv", index=False, date_format="%Y-%m-%d")
    ohlcv_factory(5).to_csv(tmp_path / "NEPSE_INDEX.csv", index=False, date_format="%Y-%m-%d")

    panel = load_panel(tmp_path)
    assert list(panel) == ["ABC"]
    assert isinstance(panel["ABC"].index, pd.DatetimeIndex)
    assert list(panel["ABC"].columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert sorted(load_panel(tmp_path, include_indices=True)) == ["ABC", "NEPSE_INDEX"]


def test_liquid_universe_ranks_by_recent_turnover_without_lookahead(ohlcv_factory):
    a = ohlcv_factory(100).set_index("timestamps")
    b = a.copy()
    b["amount"] = 1.0
    c = a.copy()
    c["amount"] = 1.0
    c.loc[c.index[80]:, "amount"] = 1e12  # only liquid AFTER the as-of date
    d = a.iloc[:70]                       # stopped trading before the as-of date
    e = a.iloc[60:]                       # only 20 rows of history by the as-of date
    panel = {"A": a, "B": b, "C": c, "D": d, "E": e}
    as_of = a.index[79]

    # B and C tie on turnover up to as_of; ties break alphabetically
    assert liquid_universe(panel, as_of, top_n=2, min_history=50) == ["A", "B"]
    assert liquid_universe(panel, as_of, top_n=10, min_history=50) == ["A", "B", "C"]


def test_rebalance_dates_every_n_market_days():
    cal = pd.bdate_range("2024-01-01", periods=30)
    assert rebalance_dates(cal, cal[3], cal[20], every=5) == [cal[3], cal[8], cal[13], cal[18]]
