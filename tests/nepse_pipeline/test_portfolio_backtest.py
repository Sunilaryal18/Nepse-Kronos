import numpy as np
import pandas as pd

from nepse_kronos.portfolio_backtest import main


def test_simple_signal_backtest_end_to_end(tmp_path, ohlcv_factory, capsys):
    clean = tmp_path / "clean"
    clean.mkdir()
    n = 300
    ohlcv_factory(n).to_csv(clean / "NEPSE_INDEX.csv", index=False, date_format="%Y-%m-%d")
    for k, growth in enumerate([-0.3, -0.1, 0.0, 0.2, 0.5, 0.8]):
        ohlcv_factory(n, close=np.linspace(100, 100 * (1 + growth), n)).to_csv(
            clean / f"S{k}.csv", index=False, date_format="%Y-%m-%d")
    dates = pd.bdate_range("2024-01-01", periods=n)
    out = tmp_path / "out"

    main(["--clean-dir", str(clean), "--signal", "momentum",
          "--start", f"{dates[100]:%Y-%m-%d}", "--end", f"{dates[250]:%Y-%m-%d}",
          "--refresh-every", "20", "--horizon", "5", "--lookback", "50", "--universe", "5",
          "--top-k", "2", "--keep-rank", "3", "--out-dir", str(out), "--tag", "test"])

    result = out / "test"
    assert list(pd.read_csv(result / "equity.csv").columns) == ["timestamps", "strategy", "benchmark"]
    assert "reason" in pd.read_csv(result / "trades.csv").columns
    assert list(pd.read_csv(result / "ic.csv").columns) == ["timestamps", "rank_ic"]
    assert (result / "equity.png").stat().st_size > 0
    assert len(list((out / "signals" / "momentum-w60").glob("*.csv"))) == 8  # rows 100, 120, ..., 240
    printed = capsys.readouterr().out
    assert "sharpe" in printed and "sold_stop_loss" in printed
