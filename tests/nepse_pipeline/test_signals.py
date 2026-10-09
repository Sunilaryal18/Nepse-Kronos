import numpy as np
import pandas as pd
import pytest

from nepse_kronos.signals import cached_signal, kronos_signal, lowvol_signal, momentum_signal, steady_signal


def _panel(ohlcv_factory, n=60):
    return {
        "UP": ohlcv_factory(n, close=np.linspace(100, 160, n)).set_index("timestamps"),
        "FLAT": ohlcv_factory(n, close=np.full(n, 100.0)).set_index("timestamps"),
        "DOWN": ohlcv_factory(n, close=np.linspace(100, 70, n)).set_index("timestamps"),
    }


def test_kronos_signal_predicted_returns_and_batching(ohlcv_factory, stub_predictor):
    panel = _panel(ohlcv_factory)
    dates = panel["UP"].index
    sig = kronos_signal(stub_predictor, panel, ["UP", "FLAT", "DOWN"], dates[39], dates[40:45],
                        lookback=20, sample_count=7, batch_size=2)

    up = panel["UP"]["close"]
    assert sig.name == "pred_return"
    assert list(sig.index) == ["DOWN", "FLAT", "UP"]
    assert sig["UP"] == pytest.approx(up.iloc[39] / up.iloc[20] - 1)
    assert sig["FLAT"] == pytest.approx(0.0)
    assert sig["UP"] > sig["FLAT"] > sig["DOWN"]
    assert [c["n"] for c in stub_predictor.batch_calls] == [2, 1]
    assert all(c["pred_len"] == 5 and c["sample_count"] == 7 for c in stub_predictor.batch_calls)


def test_kronos_signal_ignores_data_after_as_of(ohlcv_factory, stub_predictor):
    panel = _panel(ohlcv_factory)
    dates = panel["UP"].index
    before = kronos_signal(stub_predictor, panel, ["UP"], dates[39], dates[40:45], lookback=20)
    panel["UP"].loc[dates[40]:, "close"] = 1e9
    after = kronos_signal(stub_predictor, panel, ["UP"], dates[39], dates[40:45], lookback=20)
    pd.testing.assert_series_equal(before, after)


def test_kronos_signal_skips_short_history(ohlcv_factory, stub_predictor):
    panel = _panel(ohlcv_factory)
    dates = panel["UP"].index
    assert kronos_signal(stub_predictor, panel, ["UP"], dates[10], dates[11:16], lookback=20).empty


def test_momentum_signal(ohlcv_factory):
    panel = _panel(ohlcv_factory)
    dates = panel["UP"].index
    sig = momentum_signal(panel, ["UP", "DOWN"], dates[39], window=5)
    up = panel["UP"]["close"]
    assert sig["UP"] == pytest.approx(up.iloc[39] / up.iloc[34] - 1)
    assert sig["DOWN"] < 0


def test_cached_signal_computes_once(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return pd.Series({"AAA": 0.05, "BBB": -0.01}, name="pred_return")

    first = cached_signal(tmp_path, pd.Timestamp("2025-01-06"), compute)
    second = cached_signal(tmp_path, pd.Timestamp("2025-01-06"), compute)
    assert len(calls) == 1
    assert (tmp_path / "2025-01-06.csv").exists()
    assert second.to_dict() == pytest.approx(first.to_dict())


def _jumpy(ohlcv_factory, n=60):
    # same 60-day rise as a smooth riser, but zig-zagging on the way
    base = np.linspace(100, 130, n)
    return ohlcv_factory(n, close=base * (1 + 0.04 * (-1) ** np.arange(n))).set_index("timestamps")


def test_steady_prefers_smooth_rise_over_jumpy_rise(ohlcv_factory):
    n = 60
    panel = {"SMOOTH": ohlcv_factory(n, close=np.linspace(100, 130, n)).set_index("timestamps"),
             "JUMPY": _jumpy(ohlcv_factory, n)}
    dates = panel["SMOOTH"].index
    sig = steady_signal(panel, ["SMOOTH", "JUMPY"], dates[-1], window=40)
    assert sig.name == "pred_return"
    assert sig["SMOOTH"] > sig["JUMPY"] > 0


def test_lowvol_ranks_calm_shares_higher(ohlcv_factory):
    n = 60
    panel = {"CALM": ohlcv_factory(n, close=np.linspace(100, 110, n)).set_index("timestamps"),
             "JUMPY": _jumpy(ohlcv_factory, n)}
    dates = panel["CALM"].index
    sig = lowvol_signal(panel, ["CALM", "JUMPY"], dates[-1], window=40)
    assert sig["CALM"] > sig["JUMPY"]
    assert (sig <= 0).all()
