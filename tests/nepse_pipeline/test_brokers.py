import numpy as np
import pandas as pd
import pytest

from nepse_kronos import brokers
from nepse_kronos.brokers import (broker_features, daily_broker_stats, load_daily_stats, main,
                                  process_floorsheets, read_floorsheet)

HEADER = "transaction,symbol,buyer,seller,quantity,rate,amount,date\n"


def _trades(date, rows):
    """rows: (symbol, buyer, seller, quantity) -> one day's floorsheet frame."""
    return pd.DataFrame([{"transaction": f"{date}-{i}", "symbol": s, "buyer": b, "seller": se,
                          "quantity": q, "rate": 100.0, "amount": 100.0 * q, "date": date}
                         for i, (s, b, se, q) in enumerate(rows)])


def _write_day(folder, date, rows):
    folder.mkdir(parents=True, exist_ok=True)
    _trades(date, rows).to_csv(folder / f"floorsheet_{date}.csv", index=False)


def _stats(days):
    """days: {date: rows} -> concatenated daily stats."""
    return pd.concat([daily_broker_stats(_trades(d, rows), date=d) for d, rows in days.items()],
                     ignore_index=True)


def _row(feats, date, symbol):
    return feats[(feats["date"] == pd.Timestamp(date)) & (feats["symbol"] == symbol)].iloc[0]


def test_daily_broker_stats_aggregates_per_symbol_and_broker():
    df = _trades("2024-01-01", [("NABIL", 1, 2, 100), ("NABIL", 1, 3, 50), ("NABIL", 2, 1, 30),
                                ("ADBL", 5, 6, 10)])
    out = daily_broker_stats(df, date="2024-01-01")
    nabil = out[out["symbol"] == "NABIL"].set_index("broker")
    assert nabil.loc[1, ["bought", "sold", "net", "buy_trades"]].tolist() == [150, 30, 120, 2]
    assert nabil.loc[2, ["bought", "sold", "net", "buy_trades"]].tolist() == [30, 100, -70, 1]
    assert nabil.loc[3, ["bought", "sold", "net", "buy_trades"]].tolist() == [0, 50, -50, 0]
    assert nabil["bought"].sum() == nabil["sold"].sum() == 180
    assert nabil["net"].sum() == 0
    assert (out["date"] == "2024-01-01").all()
    assert set(out["symbol"]) == {"NABIL", "ADBL"}


def test_read_floorsheet_drops_malformed_rows(tmp_path):
    path = tmp_path / "floorsheet_2024-01-01.csv"
    path.write_text(HEADER
                    + "1,NABIL,1,2,100,500,50000,2024-01-01\n"
                    + "<<<<<<< HEAD\n"
                    + "2,NABIL,1,2,abc,500,0,2024-01-01\n"
                    + "=======\n"
                    + "1,NABIL,1,2,100,500,50000,2024-01-01\n"
                    + ">>>>>>> abc123\n"
                    + "3,adbl,4,5,7,300,2100,2024-01-01\n")
    df = read_floorsheet(path)
    assert sorted(df["symbol"]) == ["ADBL", "NABIL"]
    assert df["quantity"].sum() == 107


def test_hhi_one_broker_buying_everything_is_one_and_two_equal_is_half():
    feats = broker_features(_stats({"2024-01-01": [("A", 1, 2, 50), ("A", 1, 3, 50)]}))
    r = _row(feats, "2024-01-01", "A")
    assert r["buyer_hhi"] == pytest.approx(1.0)
    assert r["seller_hhi"] == pytest.approx(0.5)
    assert r["trades"] == 2
    assert r["top5_net_buy"] == pytest.approx(1.0)


def test_hhi_hand_computed_unequal_shares():
    # buyers 1:60, 2:30, 3:10 of 100 -> 0.36 + 0.09 + 0.01 = 0.46
    feats = broker_features(_stats({"2024-01-01": [("A", 1, 9, 60), ("A", 2, 9, 30), ("A", 3, 9, 10)]}))
    r = _row(feats, "2024-01-01", "A")
    assert r["buyer_hhi"] == pytest.approx(0.46)
    assert r["seller_hhi"] == pytest.approx(1.0)


def test_top5_net_buy_uses_five_largest_positive_net_buyers():
    # 7 brokers each net-buy 10*k units from seller 99: nets 10..70, total 280. top5 = 30+..+70 = 250.
    rows = [("A", k, 99, 10 * k) for k in range(1, 8)]
    r = _row(broker_features(_stats({"2024-01-01": rows})), "2024-01-01", "A")
    assert r["top5_net_buy"] == pytest.approx(250 / 280)


def test_top5_net_buy_with_fewer_than_five_net_buyers():
    # broker 1 buys 100 from 2; broker 2 buys 40 from 3 -> nets: 1:+100, 2:-60, 3:-40. total 140.
    rows = [("A", 1, 2, 100), ("A", 2, 3, 40)]
    r = _row(broker_features(_stats({"2024-01-01": rows})), "2024-01-01", "A")
    assert r["top5_net_buy"] == pytest.approx(100 / 140)


def test_top5_net_buy_is_zero_when_nobody_net_buys():
    # broker 1 buys and sells the same amount to itself-like pair -> all nets zero
    rows = [("A", 1, 2, 50), ("A", 2, 1, 50)]
    r = _row(broker_features(_stats({"2024-01-01": rows})), "2024-01-01", "A")
    assert r["top5_net_buy"] == 0.0


def test_rolling_window_uses_only_past_days():
    days = {"2024-01-01": [("A", 1, 2, 100)],
            "2024-01-02": [("A", 1, 2, 100)],
            "2024-01-03": [("A", 3, 4, 200)]}
    stats = _stats(days)
    full = broker_features(stats, window=2)
    truncated = broker_features(stats[stats["date"] <= "2024-01-02"], window=2)
    pd.testing.assert_frame_equal(full[full["date"] <= "2024-01-02"].reset_index(drop=True),
                                  truncated.reset_index(drop=True))
    assert _row(full, "2024-01-02", "A")["buyer_hhi"] == pytest.approx(1.0)
    # window=2 on 01-03 covers 01-02 and 01-03: buyers 1:100, 3:200
    r = _row(full, "2024-01-03", "A")
    assert r["buyer_hhi"] == pytest.approx((1 / 3) ** 2 + (2 / 3) ** 2)
    assert r["trades"] == 2
    assert r["top5_net_buy"] == pytest.approx(1.0)


def test_window_counts_the_symbols_own_trading_days():
    days = {"2024-01-01": [("A", 1, 2, 10), ("B", 5, 6, 10)],
            "2024-01-02": [("B", 5, 6, 10)],
            "2024-01-03": [("A", 3, 2, 10), ("B", 5, 6, 10)]}
    feats = broker_features(_stats(days), window=2)
    assert len(feats[feats["symbol"] == "A"]) == 2
    assert _row(feats, "2024-01-03", "A")["buyer_hhi"] == pytest.approx(0.5)
    assert _row(feats, "2024-01-03", "B")["trades"] == 2


def test_incremental_cache_skips_processed_days(tmp_path, monkeypatch):
    src, cache = tmp_path / "floorsheet", tmp_path / "cache"
    _write_day(src, "2024-01-01", [("A", 1, 2, 10)])
    _write_day(src, "2024-01-02", [("A", 1, 2, 20)])
    processed, malformed = process_floorsheets(src, cache, log=lambda *_: None)
    assert processed == ["2024-01-01", "2024-01-02"] and malformed == []

    calls = []
    real = brokers.read_floorsheet
    monkeypatch.setattr(brokers, "read_floorsheet", lambda p: calls.append(p.name) or real(p))
    _write_day(src, "2024-01-03", [("A", 3, 2, 30)])
    processed, _ = process_floorsheets(src, cache, log=lambda *_: None)
    assert processed == ["2024-01-03"]
    assert calls == ["floorsheet_2024-01-03.csv"]
    stats = load_daily_stats(cache)
    assert sorted(stats["date"].unique()) == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_main_writes_feature_file(tmp_path):
    repo = tmp_path / "repo"
    _write_day(repo / "floorsheet", "2024-01-01", [("A", 1, 2, 10), ("A", 3, 2, 10)])
    out = tmp_path / "meta" / "broker_features.csv.gz"
    assert main(["--repo-dir", str(repo), "--cache-dir", str(tmp_path / "cache"),
                 "--out", str(out), "--no-sync"]) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["date", "symbol", "top5_net_buy", "buyer_hhi", "seller_hhi", "trades"]
    assert df.iloc[0].tolist() == ["2024-01-01", "A", 1.0, 0.5, 1.0, 2]
    assert np.isfinite(df[["top5_net_buy", "buyer_hhi", "seller_hhi"]].to_numpy()).all()
