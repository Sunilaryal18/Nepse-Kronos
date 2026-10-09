import pandas as pd
import pytest

from nepse_kronos.forecast import FEATURES, forecast_next, plot_forecast


def test_uses_last_lookback_rows_and_nepse_future_dates(ohlcv_factory, stub_predictor):
    df = ohlcv_factory(50, start="2024-01-01")  # last row is Fri 2024-03-08
    pred = forecast_next(stub_predictor, df, pred_len=3, lookback=20,
                         holidays=[pd.Timestamp("2024-03-12")], sample_count=4)

    call = stub_predictor.calls[0]
    assert len(call["df"]) == 20
    assert list(call["df"].columns) == FEATURES
    assert call["x_timestamp"].iloc[-1] == pd.Timestamp("2024-03-08")
    assert call["df"]["close"].iloc[-1] == df["close"].iloc[-1]
    assert call["sample_count"] == 4
    assert pred.index.strftime("%Y-%m-%d").tolist() == ["2024-03-11", "2024-03-13", "2024-03-14"]
    assert pred.index.name == "timestamps"


def test_rejects_too_little_history(ohlcv_factory, stub_predictor):
    with pytest.raises(ValueError, match="at least 30"):
        forecast_next(stub_predictor, ohlcv_factory(10), pred_len=5, lookback=30)


def test_plot_writes_png(ohlcv_factory, stub_predictor, tmp_path):
    df = ohlcv_factory(40)
    pred = forecast_next(stub_predictor, df, pred_len=5, lookback=30)
    path = tmp_path / "chart.png"
    plot_forecast(df, pred, path, "TEST")
    assert path.stat().st_size > 0


def test_load_predictor_puts_models_in_eval_mode(monkeypatch):
    # Kronos.from_pretrained returns modules in training mode, which leaves dropout on during forecasting.
    import torch.nn as nn

    import nepse_kronos.forecast as forecast

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(0.5)

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    monkeypatch.setattr(forecast, "Kronos", Dummy)
    monkeypatch.setattr(forecast, "KronosTokenizer", Dummy)
    predictor = forecast.load_predictor(device="cpu")
    assert not predictor.model.training
    assert not predictor.tokenizer.training
