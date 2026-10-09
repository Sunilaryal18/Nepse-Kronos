import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n, start="2024-01-01", close=None):
    """Synthetic daily canonical OHLCV frame on Mon-Fri dates."""
    ts = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 100.0 + n - 1, n) if close is None else np.asarray(close, dtype=float)
    return pd.DataFrame({
        "timestamps": ts,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
        "amount": close * 1000.0,
    })


class StubPredictor:
    """Stands in for KronosPredictor: repeats the last input row for every future step."""

    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.calls.append({"df": df, "x_timestamp": x_timestamp, "y_timestamp": y_timestamp,
                           "pred_len": pred_len, **kwargs})
        last = df.iloc[-1].to_numpy()
        return pd.DataFrame([last] * pred_len, columns=df.columns, index=pd.DatetimeIndex(y_timestamp))


@pytest.fixture
def ohlcv_factory():
    return make_ohlcv


@pytest.fixture
def stub_predictor():
    return StubPredictor()
