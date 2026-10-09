import numpy as np
import pandas as pd
import pytest

from nepse_kronos.investor_backtest import main


def _market(tmp_path, ohlcv_factory, n=260):
    clean, meta = tmp_path / "clean_full", tmp_path / "meta"
    clean.mkdir()
    meta.mkdir()
    ohlcv_factory(n).to_csv(clean / "NEPSE_INDEX.csv", index=False, date_format="%Y-%m-%d")
    for k in range(6):
        close = 100 * (1 + 0.002 * k * (-1) ** np.arange(n)) + np.arange(n) * 0.05
        ohlcv_factory(n, close=close).to_csv(clean / f"S{k}.csv", index=False, date_format="%Y-%m-%d")
    days = pd.bdate_range("2024-01-01", periods=n)
    pd.DataFrame({"month": days[:8:2], "interbank": 3.0, "cd_ratio": 80.0, "lending_rate": 9.0,
                  "available_from": days[:8:2]}).to_csv(meta / "macro_monthly.csv", index=False)
    pd.DataFrame({"symbol": ["S0"], "eps": [-1.0], "net_worth_per_share": [100.0],
                  "submitted": [days[10]]}).to_csv(meta / "fundamentals.csv", index=False)
    pd.DataFrame({"date": [days[100]] * 6, "symbol": [f"S{k}" for k in range(6)],
                  "top5_net_buy": [0.1, 0.5, 0.1, 0.1, 0.1, 0.1], "buyer_hhi": 0.05, "seller_hhi": 0.05,
                  "trades": 100}).to_csv(meta / "broker_features.csv.gz", index=False)
    pd.DataFrame({"symbol": ["S1"], "ex_date": [days[150]], "cash_per_unit": [5.0]}).to_csv(
        meta / "cash_dividends.csv", index=False)
    pd.DataFrame({"symbol": [f"S{k}" for k in range(6)], "name": "x",
                  "sector": ["Bank", "Bank", "Bank", "Hydro", "Hydro", "Mutual Fund"], "listed": True}).to_csv(
        meta / "sectors.csv", index=False)
    return clean, meta, days


def test_all_features_run_end_to_end(tmp_path, ohlcv_factory, capsys):
    clean, meta, days = _market(tmp_path, ohlcv_factory)
    out = tmp_path / "out"
    main(["--clean-dir", str(clean), "--meta-dir", str(meta), "--out-dir", str(out),
          "--start", f"{days[60]:%Y-%m-%d}", "--universe", "6", "--top-k", "3", "--keep-rank", "4",
          "--min-history", "50", "--features", "money,quality,broker,sectors,taxwait", "--tag", "all"])
    trades = pd.read_csv(out / "all" / "trades.csv")
    assert "S0" not in set(trades.loc[trades.side == "buy", "symbol"])      # loss-maker blocked by quality
    assert "S5" not in set(trades["symbol"])                                # mutual funds are not traded
    assert list(pd.read_csv(out / "all" / "equity.csv").columns) == ["timestamps", "strategy", "benchmark"]
    assert "features: money,quality,broker,sectors,taxwait" in capsys.readouterr().out


def test_dividends_feature_is_refused_because_prices_already_include_them(tmp_path, ohlcv_factory):
    clean, meta, days = _market(tmp_path, ohlcv_factory)
    with pytest.raises(SystemExit):
        main(["--clean-dir", str(clean), "--meta-dir", str(meta), "--out-dir", str(tmp_path / "out"),
              "--start", f"{days[60]:%Y-%m-%d}", "--features", "dividends"])
