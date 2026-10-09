import pandas as pd
import pytest

from nepse_kronos.history import adjust_prices, adjustment_factors, cash_per_unit, corporate_events, load_raw_ohlcv


def _raw(closes, start="2023-09-25"):
    days = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                       "volume": 1000.0, "amount": [c * 1000.0 for c in closes]}, index=days)
    df.index.name = "timestamps"
    return df


def _events(rows):
    return pd.DataFrame(rows, columns=["book_close", "cash", "bonus_pct", "right_ratio", "right_price"]).assign(
        book_close=lambda d: pd.to_datetime(d["book_close"]))


def test_bonus_and_cash_like_nica():
    # NICA FY 2079/80: bonus 29%, cash 1.52%, book closure 2023-10-03; close before was 761.5
    df = _raw([777.4, 765.0, 761.5, 573.0, 568.0], start="2023-09-28")
    events = _events([["2023-10-03", 1.52, 29.0, 0.0, 100.0]])
    with_cash = adjustment_factors(df, events, include_cash=True)
    assert with_cash.iloc[2] * 761.5 == pytest.approx(589.1293, rel=1e-5)   # matches the published adjusted price
    assert with_cash.iloc[3:].tolist() == [1.0, 1.0]
    without_cash = adjustment_factors(df, events, include_cash=False)
    assert without_cash.iloc[0] == pytest.approx(1 / 1.29)


def test_right_share_uses_nepse_formula():
    df = _raw([200.0, 200.0, 150.0, 150.0])
    events = _events([[df.index[2], 0.0, 0.0, 1.0, 100.0]])   # 1:1 right at Rs 100
    assert adjustment_factors(df, events).tolist() == pytest.approx([0.75, 0.75, 1.0, 1.0])


def test_book_close_on_holiday_counts_from_next_trading_day_and_events_compound():
    df = _raw([100.0] * 6)                                    # Mon 2023-09-25 .. Mon 2023-10-02
    events = _events([["2023-09-30", 0.0, 100.0, 0.0, 100.0],  # Saturday -> applies from Monday 10-02
                      ["2023-09-27", 0.0, 25.0, 0.0, 100.0]])
    assert adjustment_factors(df, events).tolist() == pytest.approx([0.4, 0.4, 0.5, 0.5, 0.5, 1.0])


def test_events_outside_the_data_are_ignored():
    df = _raw([100.0] * 3)
    events = _events([["2020-01-01", 0.0, 50.0, 0.0, 100.0], ["2030-01-01", 0.0, 50.0, 0.0, 100.0]])
    assert adjustment_factors(df, events).tolist() == [1.0, 1.0, 1.0]


def test_adjust_prices_scales_prices_and_volume_not_amount():
    df = _raw([100.0, 100.0, 50.0, 50.0])
    events = _events([[df.index[2], 0.0, 100.0, 0.0, 100.0]])
    out = adjust_prices(df, events)
    assert out["close"].tolist() == [50.0] * 4
    assert out["volume"].tolist() == [2000.0, 2000.0, 1000.0, 1000.0]
    assert out["amount"].tolist() == df["amount"].tolist()


def test_cash_per_unit_in_todays_share_terms():
    df = _raw([100.0] * 6)
    events = _events([[df.index[1], 10.0, 0.0, 0.0, 100.0],      # Rs 10 cash (10% of Rs 100 face value)
                      [df.index[4], 0.0, 100.0, 0.0, 100.0]])    # later 100% bonus halves each old share
    per_unit = cash_per_unit(df, events)
    assert per_unit.to_dict() == {df.index[1]: pytest.approx(5.0)}


def test_corporate_events_merges_dividends_and_rights():
    actions = pd.DataFrame({"symbol": ["A", "A", "B"], "fiscal_year": ["x", "y", "z"],
                            "cash_pct": [10.0, None, 0.0], "bonus_pct": [5.0, 20.0, 0.0],
                            "book_close": ["2023-01-02", None, "2023-05-01"]})
    rights = pd.DataFrame({"symbol": ["A"], "ratio_new_per_old": [0.5], "book_close": ["2023-01-02"]})
    ev = corporate_events(actions, rights)
    a = ev[ev["symbol"] == "A"]
    assert len(a) == 1                                          # the row without a book close is dropped
    assert a[["cash", "bonus_pct", "right_ratio"]].values.tolist() == [[10.0, 5.0, 0.5]]
    assert ev[ev["symbol"] == "B"].empty                        # nothing happened


def test_load_raw_ohlcv(tmp_path):
    day_dir = tmp_path / "2023"
    day_dir.mkdir()
    (day_dir / "2023-01-02.csv").write_text("date,symbol,open,high,low,close,volume,turnover,trades\n"
                                            "2023-01-02,AAA,10,11,9,10.5,100,1050,3\n")
    (day_dir / "2023-01-03.csv").write_text("date,symbol,open,high,low,close,volume,turnover,trades\n")
    panel = load_raw_ohlcv(tmp_path)
    assert list(panel) == ["AAA"]
    assert panel["AAA"].loc["2023-01-02", "amount"] == 1050


def test_file_names_are_safe_for_symbols_with_slashes():
    from nepse_kronos.history import file_name
    assert file_name("GBD80/81") == "GBD80_81.csv"
    assert file_name("NABIL") == "NABIL.csv"


def test_repair_gaps_fixes_drops_beyond_the_daily_price_limit():
    from nepse_kronos.history import repair_gaps
    df = _raw([100.0, 100.0, 50.0, 50.0, 46.0, 55.2])        # -50% (missed bonus), -8% (real), +20% (allowed)
    out = repair_gaps(df)
    assert out["close"].tolist() == pytest.approx([50.0, 50.0, 50.0, 50.0, 46.0, 55.2])
    assert out["volume"].iloc[0] == pytest.approx(2000.0)


def test_build_keeps_reference_series_and_adds_delisted_ones(tmp_path):
    from nepse_kronos.history import build
    reference, out = tmp_path / "clean", tmp_path / "clean_full"
    reference.mkdir()
    keep = _raw([10.0, 11.0, 12.0])
    keep.reset_index().assign(amount=1.0).to_csv(reference / "LISTED.csv", index=False, date_format="%Y-%m-%d")
    keep.reset_index().to_csv(reference / "NEPSE_INDEX.csv", index=False, date_format="%Y-%m-%d")
    raw = {"LISTED": _raw([999.0] * 3), "GONE": _raw([100.0, 100.0, 40.0, 40.0])}
    events = _events([]).assign(symbol=[])
    build(raw, events, reference, out)
    assert pd.read_csv(out / "LISTED.csv")["close"].tolist() == [10.0, 11.0, 12.0]   # untouched
    assert pd.read_csv(out / "GONE.csv")["close"].tolist() == pytest.approx([40.0] * 4)   # added and repaired
    assert (out / "NEPSE_INDEX.csv").exists()


def test_fix_stale_adjustments_applies_only_missing_events():
    from nepse_kronos.history import fix_stale_adjustments
    days = pd.bdate_range("2025-01-06", periods=8)
    official = pd.Series([200.0, 200.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0], index=days)   # two 100% bonuses
    # the reference applied the first bonus (day 2) but missed the second (day 5)
    reference = pd.DataFrame({c: [50.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 50.0] for c in ["open", "high", "low", "close"]},
                             index=days).assign(volume=1000.0, amount=1.0)
    reference.iloc[2, :4] = 50.0
    reference.loc[days[:2], ["open", "high", "low", "close"]] = 100.0
    reference.loc[days[2:5], ["open", "high", "low", "close"]] = 100.0
    events = _events([[days[2], 0.0, 100.0, 0.0, 100.0], [days[5], 0.0, 100.0, 0.0, 100.0]])
    fixed, applied = fix_stale_adjustments(reference, official, events)
    assert applied == [days[5]]
    assert fixed["close"].tolist() == pytest.approx([50.0] * 8)
    assert fixed["volume"].iloc[0] == pytest.approx(2000.0)


def test_build_fixes_stale_listed_series_when_official_prices_given(tmp_path):
    from nepse_kronos.history import build
    reference, out = tmp_path / "clean", tmp_path / "clean_full"
    reference.mkdir()
    days = pd.bdate_range("2025-01-06", periods=4)
    stale = pd.DataFrame({"timestamps": days, "open": [100.0, 100.0, 50.0, 50.0], "high": [100.0, 100.0, 50.0, 50.0],
                          "low": [100.0, 100.0, 50.0, 50.0], "close": [100.0, 100.0, 50.0, 50.0],
                          "volume": 1000.0, "amount": 1.0})
    stale.to_csv(reference / "LISTED.csv", index=False, date_format="%Y-%m-%d")
    official = {"LISTED": pd.Series([100.0, 100.0, 50.0, 50.0], index=days)}
    events = _events([[days[2], 0.0, 100.0, 0.0, 100.0]]).assign(symbol="LISTED")
    build({}, events, reference, out, official=official)
    assert pd.read_csv(out / "LISTED.csv")["close"].tolist() == pytest.approx([50.0] * 4)


def test_build_appends_newer_live_days_only(tmp_path):
    from nepse_kronos.history import build
    reference, out, live = tmp_path / "clean", tmp_path / "clean_full", tmp_path / "live"
    reference.mkdir()
    live.mkdir()
    days = pd.bdate_range("2026-09-30", periods=5)                 # Wed .. Tue
    base = pd.DataFrame({"timestamps": days[:3], "open": 10.0, "high": 10.0, "low": 10.0, "close": [10.0, 11.0, 12.0],
                         "volume": 1.0, "amount": 1.0})
    base.to_csv(reference / "LISTED.csv", index=False, date_format="%Y-%m-%d")
    base.to_csv(reference / "NEPSE_INDEX.csv", index=False, date_format="%Y-%m-%d")
    newer = pd.DataFrame({"timestamps": days[2:], "open": 12.0, "high": 13.0, "low": 12.0, "close": [99.0, 13.0, 14.0],
                          "volume": 5.0, "amount": 50.0})                # first row overlaps: ignored
    newer.to_csv(live / "LISTED.csv", index=False, date_format="%Y-%m-%d")
    newer.to_csv(live / "NEPSE_INDEX.csv", index=False, date_format="%Y-%m-%d")
    build({}, _events([]).assign(symbol=[]), reference, out, live_dir=live)
    for name in ["LISTED.csv", "NEPSE_INDEX.csv"]:
        assert pd.read_csv(out / name)["close"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
