from pathlib import Path

import pandas as pd


def load_panel(clean_dir, include_indices=False):
    """All clean CSVs as {symbol: DataFrame indexed by date}."""
    panel = {}
    for path in sorted(Path(clean_dir).glob("*.csv")):
        symbol = path.stem
        if symbol.endswith("_INDEX") and not include_indices:
            continue
        panel[symbol] = pd.read_csv(path, parse_dates=["timestamps"]).set_index("timestamps")
    return panel


def liquid_universe(panel, as_of, top_n, min_history=150, liquidity_window=60):
    """Most-traded symbols that traded on as_of, using only data up to as_of."""
    as_of = pd.Timestamp(as_of)
    ranked = []
    for symbol, df in panel.items():
        history = df.loc[:as_of]
        if len(history) < min_history or history.index[-1] != as_of:
            continue
        ranked.append((symbol, history["amount"].iloc[-liquidity_window:].median()))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _ in ranked[:top_n]]


def rebalance_dates(calendar, start, end, every):
    """Every `every`-th market day between start and end (inclusive)."""
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return list(days[::every])
