import json

import numpy as np

from nepse_kronos.paper_trade import main


def _write_market(clean, ohlcv_factory, n):
    clean.mkdir(exist_ok=True)
    ohlcv_factory(n).to_csv(clean / "NEPSE_INDEX.csv", index=False, date_format="%Y-%m-%d")
    for k, wiggle in enumerate([0.0, 0.002, 0.004, 0.006, 0.008, 0.010]):
        close = 100 * (1 + wiggle * (-1) ** np.arange(n)) + np.arange(n) * 0.1
        ohlcv_factory(n, close=close).to_csv(clean / f"S{k}.csv", index=False, date_format="%Y-%m-%d")


def test_first_run_plans_buys_then_later_run_holds_them(tmp_path, ohlcv_factory, capsys):
    clean, out = tmp_path / "clean", tmp_path / "paper"
    args = ["--clean-dir", str(clean), "--out-dir", str(out), "--no-update",
            "--min-history", "50", "--universe", "5", "--top-k", "2", "--keep-rank", "3", "--features", "",
            "--meta-dir", str(tmp_path / "no-meta")]

    _write_market(clean, ohlcv_factory, 200)
    main(args)
    first = capsys.readouterr().out
    config = json.loads((out / "config.json").read_text())
    assert config["start"] == "2024-10-04"          # the 200th weekday from 2024-01-01
    assert config["signal"] == "lowvol" and config["stop_loss"] is None
    assert first.count("BUY") == 2
    assert "S0" in first and "S1" in first          # the two calmest shares
    assert (out / "reports" / "2024-10-04.md").exists()

    _write_market(clean, ohlcv_factory, 203)        # three more trading days arrive
    main(args)
    later = capsys.readouterr().out
    assert json.loads((out / "config.json").read_text())["start"] == "2024-10-04"  # same paper portfolio
    assert "You own 2 shares" in later
    assert (out / "reports" / "2024-10-09.md").exists()


def test_features_are_saved_and_money_conditions_reported(tmp_path, ohlcv_factory, capsys):
    import pandas as pd
    clean, out, meta = tmp_path / "clean", tmp_path / "paper", tmp_path / "meta"
    _write_market(clean, ohlcv_factory, 200)
    meta.mkdir()
    months = pd.bdate_range("2024-01-01", periods=200)[::20]
    pd.DataFrame({"month": months, "interbank": 5.0, "cd_ratio": 80.0, "lending_rate": 9.0,
                  "available_from": months}).to_csv(meta / "macro_monthly.csv", index=False)
    pd.DataFrame({"symbol": [f"S{k}" for k in range(6)], "name": "x", "sector": ["Bank"] * 6,
                  "listed": True}).to_csv(meta / "sectors.csv", index=False)
    main(["--clean-dir", str(clean), "--out-dir", str(out), "--meta-dir", str(meta), "--no-update",
          "--min-history", "50", "--universe", "5", "--top-k", "4", "--keep-rank", "5", "--features", "money,sectors"])
    printed = capsys.readouterr().out
    assert json.loads((out / "config.json").read_text())["features"] == ["money", "sectors"]
    assert "Money conditions: invest in 2 of 4 slots" in printed   # score 0 -> 50%
    assert printed.count("BUY") == 2
