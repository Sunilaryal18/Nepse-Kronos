"""The newest trading days straight from the NEPSE website, for when the community datasets lag.

NEPSE publishes index history (open/high/low/close) and per-share history (high/low/close, volume,
turnover - but no opening price). For these few newest days a share's open is taken as the previous
close. Rows are written to data/nepse/live/<SYMBOL>.csv only for days after the community data ends;
`nepse_kronos.history` appends them, and they are ignored once the community data catches up.

Usage:
    python -m nepse_kronos.live_prices            # ~200 most-traded shares + paper-portfolio holdings
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from nepse_kronos.nepse_reports import NepseClient
from nepse_kronos.schema import CANONICAL_COLUMNS

log = logging.getLogger(__name__)
LIVE_DIR = Path("data/nepse/live")
INDEX_URL = "/api/nots/index/history/58?size={size}"
PRICE_URL = "/api/nots/market/history/security/{security_id}?&size=500&startDate={start}&endDate={end}"
LOOKBACK_DAYS = 7  # fetch a little before the gap so the first new day has a previous close


class LiveClient(NepseClient):
    def security_ids(self):
        return self._call(self._nepse.getSecurityIDKeyMap)

    def index_history(self, size):
        return self._call(self._nepse.requestGETAPI, INDEX_URL.format(size=size))

    def price_history(self, security_id, start, end):
        return self._call(self._nepse.requestGETAPI, PRICE_URL.format(security_id=security_id, start=start, end=end))


def _rows(payload):
    return payload.get("content", []) if isinstance(payload, dict) else list(payload or [])


def _frame(rows, mapping):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = df.rename(columns=mapping)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").drop_duplicates("timestamps", keep="last").reset_index(drop=True)


def parse_index_history(payload):
    df = _frame(_rows(payload), {"businessDate": "timestamps", "openIndex": "open", "highIndex": "high",
                                 "lowIndex": "low", "closingIndex": "close", "turnoverVolume": "volume",
                                 "turnoverValue": "amount"})
    return df[CANONICAL_COLUMNS] if not df.empty else df


def parse_price_history(payload):
    df = _frame(_rows(payload), {"businessDate": "timestamps", "highPrice": "high", "lowPrice": "low",
                                 "closePrice": "close", "totalTradedQuantity": "volume",
                                 "totalTradedValue": "amount"})
    if df.empty:
        return df
    df["open"] = df["close"].shift(1).fillna(df["close"])  # NEPSE doesn't publish the open here
    return df[CANONICAL_COLUMNS]


def _write_after(df, since, path):
    """Write the rows dated after `since` (deleting the file when there are none); returns their count."""
    new = df[df["timestamps"] > since] if not df.empty else df
    if new.empty:
        path.unlink(missing_ok=True)
    else:
        new.to_csv(path, index=False, date_format="%Y-%m-%d")
    return len(new)


def update_live(client, symbols, since, live_dir=LIVE_DIR, today=None):
    """Write the trading days after `since` for the NEPSE index and each symbol; returns {symbol: new rows}."""
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    since = pd.Timestamp(since)

    index = parse_index_history(client.index_history(size=30))
    written = {"NEPSE_INDEX": _write_after(index, since, live_dir / "NEPSE_INDEX.csv")}

    ids = client.security_ids()
    start = f"{since - pd.Timedelta(days=LOOKBACK_DAYS):%Y-%m-%d}"
    for symbol in symbols:
        if symbol not in ids:
            log.warning("no NEPSE security id for %s", symbol)
            continue
        prices = parse_price_history(client.price_history(ids[symbol], start, f"{today:%Y-%m-%d}"))
        written[symbol] = _write_after(prices, since, live_dir / f"{symbol}.csv")
    return written


def symbols_to_fetch(clean_dir, top=200, also_from=None):
    """The most-traded shares (median turnover, last 60 days) plus any symbol in a trades file."""
    turnover = {}
    for path in Path(clean_dir).glob("*.csv"):
        if path.stem.endswith("_INDEX"):
            continue
        tail = pd.read_csv(path, usecols=["amount"]).tail(60)
        turnover[path.stem] = tail["amount"].median()
    chosen = set(sorted(turnover, key=lambda s: -(turnover[s] or 0))[:top])
    if also_from and Path(also_from).exists():
        chosen |= set(pd.read_csv(also_from)["symbol"].dropna())
    return sorted(chosen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", default="data/nepse/clean", help="community data; new days start after it")
    parser.add_argument("--live-dir", default=str(LIVE_DIR))
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--also-from", default="outputs/nepse/paper/trades.csv")
    parser.add_argument("--min-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy

    since = pd.read_csv(Path(args.clean_dir) / "NEPSE_INDEX.csv", parse_dates=["timestamps"])["timestamps"].max()
    symbols = symbols_to_fetch(args.clean_dir, args.top, args.also_from)
    written = update_live(LiveClient(min_interval=max(args.min_interval, 1.0)), symbols, since, args.live_dir)
    days = written.get("NEPSE_INDEX", 0)
    print(f"NEPSE website: {days} trading day(s) after {since:%Y-%m-%d}; "
          f"{sum(1 for s, n in written.items() if s != 'NEPSE_INDEX' and n)} of {len(symbols)} shares updated")


if __name__ == "__main__":
    main()
