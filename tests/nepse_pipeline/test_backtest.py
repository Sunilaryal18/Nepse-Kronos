import pandas as pd
import pytest

from nepse_kronos.backtest import summarize, walk_forward


def test_windows_cover_history_without_lookahead(ohlcv_factory, stub_predictor):
    df = ohlcv_factory(30)
    results = walk_forward(stub_predictor, df, lookback=10, pred_len=5, step=5)

    # origins end at rows 9, 14, 19, 24 (row 29 has no 5 future rows left)
    assert len(results) == 4
    for call, row in zip(stub_predictor.calls, results.itertuples()):
        assert call["x_timestamp"].iloc[-1] == row.origin
        assert call["y_timestamp"].iloc[0] > row.origin
        assert len(call["df"]) == 10
    assert results["origin"].iloc[0] == df["timestamps"].iloc[9]
    assert results["target"].iloc[0] == df["timestamps"].iloc[14]
    # the stub repeats the last close, so its predicted return is zero
    assert (results["pred_return"] == 0).all()
    assert results["actual_return"].iloc[0] == pytest.approx(df["close"].iloc[14] / df["close"].iloc[9] - 1)


def test_start_date_skips_earlier_origins(ohlcv_factory, stub_predictor):
    df = ohlcv_factory(30)
    results = walk_forward(stub_predictor, df, lookback=10, pred_len=5, step=5,
                           start_date=df["timestamps"].iloc[15])
    assert results["origin"].tolist() == [df["timestamps"].iloc[19], df["timestamps"].iloc[24]]


def test_summarize_metrics():
    results = pd.DataFrame({
        "last_close": [100.0, 100.0, 100.0, 100.0],
        "pred_close": [110.0, 90.0, 105.0, 95.0],
        "actual_close": [120.0, 80.0, 95.0, 100.0],
        "pred_return": [0.10, -0.10, 0.05, -0.05],
        "actual_return": [0.20, -0.20, -0.05, 0.0],
    })
    s = summarize(results)
    assert s["windows"] == 4
    assert s["direction_accuracy"] == pytest.approx(0.5)
    assert s["mape"] == pytest.approx((10 / 120 + 10 / 80 + 10 / 95 + 5 / 100) / 4)
    assert s["naive_mape"] == pytest.approx((20 / 120 + 20 / 80 + 5 / 95 + 0) / 4)
    assert -1.0 <= s["rank_ic"] <= 1.0
