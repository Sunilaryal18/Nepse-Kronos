"""Rankings of shares on a date. Every signal returns a Series named pred_return: higher = better."""
from pathlib import Path

import numpy as np
import pandas as pd

from nepse_kronos.forecast import FEATURES

SIMPLE_SIGNALS = {"momentum", "steady", "lowvol"}


def _series(values):
    return pd.Series(values, name="pred_return", dtype=float).sort_index()


def kronos_signal(predictor, panel, symbols, as_of, future_dates, lookback=128, sample_count=10, batch_size=32):
    """Kronos-predicted return from the as_of close to the close on the last future date."""
    as_of = pd.Timestamp(as_of)
    future = pd.Series(pd.DatetimeIndex(future_dates))
    ready = [(s, panel[s].loc[:as_of].iloc[-lookback:]) for s in symbols]
    ready = [(s, h) for s, h in ready if len(h) == lookback]
    predictions = {}
    for start in range(0, len(ready), batch_size):
        chunk = ready[start:start + batch_size]
        outputs = predictor.predict_batch(
            df_list=[h[FEATURES].reset_index(drop=True) for _, h in chunk],
            x_timestamp_list=[pd.Series(h.index) for _, h in chunk],
            y_timestamp_list=[future for _ in chunk],
            pred_len=len(future),
            T=1.0,
            top_p=0.9,
            sample_count=sample_count,
            verbose=False,
        )
        for (symbol, history), out in zip(chunk, outputs):
            predictions[symbol] = float(out["close"].iloc[-1]) / history["close"].iloc[-1] - 1.0
    return _series(predictions)


def _recent_closes(panel, symbols, as_of, window):
    as_of = pd.Timestamp(as_of)
    for symbol in symbols:
        close = panel[symbol].loc[:as_of, "close"]
        if len(close) > window:
            yield symbol, close.iloc[-1 - window:]


def momentum_signal(panel, symbols, as_of, window=60):
    """Biggest rise over the last `window` days."""
    return _series({s: c.iloc[-1] / c.iloc[0] - 1.0 for s, c in _recent_closes(panel, symbols, as_of, window)})


def steady_signal(panel, symbols, as_of, window=60):
    """Rise over the last `window` days divided by how jumpy the share was over the same days."""
    values = {}
    for symbol, close in _recent_closes(panel, symbols, as_of, window):
        swing = close.pct_change().std() * np.sqrt(window)
        if swing > 0:
            values[symbol] = (close.iloc[-1] / close.iloc[0] - 1.0) / swing
    return _series(values)


def lowvol_signal(panel, symbols, as_of, window=60):
    """Calmest shares first (negative daily volatility)."""
    return _series({s: -c.pct_change().std() for s, c in _recent_closes(panel, symbols, as_of, window)})


def cached_signal(cache_dir, as_of, compute):
    """Load the signal for as_of from cache_dir, or compute and save it."""
    path = Path(cache_dir) / f"{pd.Timestamp(as_of):%Y-%m-%d}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)["pred_return"]
    signal = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    signal.rename_axis("symbol").to_frame("pred_return").to_csv(path)
    return signal
