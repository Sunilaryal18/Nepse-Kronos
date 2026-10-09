import pandas as pd

CANONICAL_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Lower-case, underscore-separated header names seen in NEPSE exports, per canonical column.
COLUMN_ALIASES = {
    "timestamps": ["timestamps", "timestamp", "date", "business_date", "businessdate", "trade_date"],
    "open": ["open", "open_price", "openprice"],
    "high": ["high", "high_price", "highprice", "max_price", "maxprice"],
    "low": ["low", "low_price", "lowprice", "min_price", "minprice"],
    "close": ["close", "close_price", "closeprice", "ltp", "last_traded_price"],
    "volume": ["volume", "qty", "quantity", "traded_shares", "total_traded_quantity",
               "totaltradedquantity", "total_traded_shares"],
    "amount": ["amount", "turnover", "total_turnover", "total_traded_value", "totaltradedvalue"],
}


def _key(name):
    return str(name).lstrip("﻿").strip().lower().replace(" ", "_").replace("(", "").replace(")", "")


def _to_number(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def normalize_ohlcv(raw):
    """Map a raw NEPSE price export onto CANONICAL_COLUMNS and drop unusable rows."""
    lookup = {_key(c): c for c in raw.columns}
    df = pd.DataFrame()
    for canon, aliases in COLUMN_ALIASES.items():
        source = next((lookup[a] for a in aliases if a in lookup), None)
        if source is None:
            if canon in ("volume", "amount"):
                continue
            raise ValueError(f"Missing required column '{canon}'. Columns found: {list(raw.columns)}")
        df[canon] = raw[source].to_numpy()

    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.normalize()
    for col in df.columns.drop("timestamps"):
        df[col] = _to_number(df[col])
    if "volume" not in df:
        df["volume"] = 0.0
    if "amount" not in df:
        df["amount"] = df["volume"] * df[PRICE_COLUMNS].mean(axis=1)

    df = df.dropna(subset=PRICE_COLUMNS)
    df = df[(df[PRICE_COLUMNS] > 0).all(axis=1)].copy()
    df[["volume", "amount"]] = df[["volume", "amount"]].fillna(0.0)
    df["high"] = df[PRICE_COLUMNS].max(axis=1)
    df["low"] = df[PRICE_COLUMNS].min(axis=1)

    df = df.drop_duplicates("timestamps", keep="last").sort_values("timestamps").reset_index(drop=True)
    return df[CANONICAL_COLUMNS].astype({c: float for c in CANONICAL_COLUMNS[1:]})


def trim_close_only_history(df):
    """Drop the leading stretch of close-only rows (open == high == low == close).

    Old NEPSE index data (before late 2016) records only the close, and Kronos was trained on real
    candles; windows built from such rows produce wild forecasts. Later flat rows are genuine
    thin-trading days and are kept.
    """
    real = ~df[PRICE_COLUMNS].eq(df["close"], axis=0).all(axis=1)
    if not real.any():
        return df
    return df.loc[real.idxmax():].reset_index(drop=True)
