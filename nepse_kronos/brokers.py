"""Broker-flow features from the NEPSE floorsheet (every trade with its buying and selling broker).

Source: https://github.com/socrateai-official/nepse-open-data `floorsheet/` (MIT licence),
one file per trading day from 2024-01, columns
`transaction,symbol,buyer,seller,quantity,rate,amount,date` (buyer/seller are broker numbers).

Pipeline:
1. `sync_floorsheet` keeps a separate sparse clone holding only the floorsheet folder.
2. Each day's file is reduced by `daily_broker_stats` to per-(symbol, broker) bought/sold units and
   cached as `floorsheet_daily/YYYY-MM-DD.csv.gz`; days already cached are skipped on re-runs.
3. `broker_features` rolls the cached days over the last `window` trading days of each symbol
   (days on which that symbol traded) and writes, per (date, symbol):
   - top5_net_buy: net units of the 5 biggest net-buying brokers / total units traded (0 when nobody
     net-buys)
   - buyer_hhi / seller_hhi: Herfindahl concentration of bought / sold units across brokers (0-1)
   - trades: number of transactions
   Values on date d use only trades on or before d.

Usage:
    python -m nepse_kronos.brokers              # update the clone, process new days, write features
    python -m nepse_kronos.brokers --no-sync    # reuse the local clone
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_REPO = "https://github.com/socrateai-official/nepse-open-data.git"
DEFAULT_REPO_DIR = Path("data/nepse/source/floorsheet-repo")
DEFAULT_CACHE_DIR = Path("data/nepse/source/floorsheet_daily")
DEFAULT_OUT = Path("data/nepse/meta/broker_features.csv.gz")
FLOORSHEET_FOLDER = "floorsheet"
FILE_RE = re.compile(r"floorsheet_(\d{4}-\d{2}-\d{2})\.csv$")
STAT_COLUMNS = ["date", "symbol", "broker", "bought", "sold", "buy_trades"]
FEATURE_COLUMNS = ["date", "symbol", "top5_net_buy", "buyer_hhi", "seller_hhi", "trades"]


def sync_floorsheet(repo_dir=DEFAULT_REPO_DIR):
    """Clone the dataset (floorsheet folder only) or update an existing clone to the latest commit."""
    repo_dir = Path(repo_dir)
    git = ["git", "-C", str(repo_dir)]
    if (repo_dir / ".git").exists():
        subprocess.run([*git, "fetch", "--quiet", "--depth", "1", "origin", "main"], check=True)
        subprocess.run([*git, "reset", "--quiet", "--hard", "FETCH_HEAD"], check=True)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--filter=blob:none", "--sparse",
                        SOURCE_REPO, str(repo_dir)], check=True)
    subprocess.run([*git, "sparse-checkout", "set", FLOORSHEET_FOLDER], check=True)


def read_floorsheet(path):
    """Read one floorsheet CSV, dropping malformed rows (bad numbers, merge-conflict debris, duplicates)."""
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip",
                     usecols=lambda c: c in {"transaction", "symbol", "buyer", "seller", "quantity"})
    for col in ["buyer", "seller", "quantity"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["symbol", "buyer", "seller", "quantity"])
    df = df[df["quantity"] > 0]
    if "transaction" in df:
        df = df.drop_duplicates("transaction", keep="last")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["quantity"] = df["quantity"].round()
    return df.astype({"buyer": "int64", "seller": "int64", "quantity": "int64"})


def daily_broker_stats(df, date=None):
    """One day's trades -> per (date, symbol, broker): bought, sold, buy_trades, net (= bought - sold).

    Summing `bought` over brokers gives the symbol's total units traded (each trade has exactly one
    buyer), and summing `buy_trades` gives its number of transactions.
    """
    if df.empty:
        return pd.DataFrame(columns=STAT_COLUMNS + ["net"])
    if date is None:
        date = df["date"].iloc[0]
    buys = (df.groupby(["symbol", "buyer"])["quantity"].agg(bought="sum", buy_trades="size")
            .rename_axis(["symbol", "broker"]))
    sells = df.groupby(["symbol", "seller"])["quantity"].sum().rename("sold").rename_axis(["symbol", "broker"])
    out = buys.join(sells, how="outer").fillna(0).astype("int64").reset_index()
    out.insert(0, "date", pd.Timestamp(date).strftime("%Y-%m-%d"))
    out["net"] = out["bought"] - out["sold"]
    return out[STAT_COLUMNS + ["net"]].sort_values(["symbol", "broker"], ignore_index=True)


def floorsheet_files(folder):
    """{date string: path} of every floorsheet_YYYY-MM-DD.csv in `folder`."""
    files = {}
    for path in sorted(Path(folder).glob("floorsheet_*.csv")):
        m = FILE_RE.search(path.name)
        if m:
            files[m.group(1)] = path
    return files


def process_floorsheets(folder, cache_dir=DEFAULT_CACHE_DIR, log=print):
    """Reduce every not-yet-cached day to per-broker stats; returns (processed dates, malformed files)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed, malformed = [], []
    for date, path in floorsheet_files(folder).items():
        target = cache_dir / f"{date}.csv.gz"
        if target.exists():
            continue
        try:
            stats = daily_broker_stats(read_floorsheet(path), date=date)
        except Exception as exc:  # unreadable file: report, skip, retry next run
            malformed.append((path.name, str(exc)))
            log(f"malformed {path.name}: {exc}")
            continue
        if stats.empty:
            malformed.append((path.name, "no valid trades"))
        tmp = target.with_suffix(".tmp")
        stats[STAT_COLUMNS].to_csv(tmp, index=False, compression="gzip")
        tmp.replace(target)
        processed.append(date)
    return processed, malformed


def load_daily_stats(cache_dir=DEFAULT_CACHE_DIR):
    """Concatenate cached per-day broker stats."""
    paths = sorted(Path(cache_dir).glob("*.csv.gz"))
    dtypes = {"symbol": "str", "broker": "int32", "bought": "int64", "sold": "int64", "buy_trades": "int32"}
    frames = [pd.read_csv(p, dtype=dtypes) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=STAT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _rolling_sum(matrix, window):
    """Trailing sum over the last `window` rows (fewer at the start), row-wise on a 2-D array."""
    c = np.cumsum(matrix, axis=0)
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    return out


def _symbol_features(rows, window):
    dates = np.sort(rows["date"].unique())
    brokers = np.sort(rows["broker"].unique())
    di = np.searchsorted(dates, rows["date"].to_numpy())
    bi = np.searchsorted(brokers, rows["broker"].to_numpy())
    shape = (len(dates), len(brokers))
    bought = np.zeros(shape)
    sold = np.zeros(shape)
    trades = np.zeros(len(dates))
    np.add.at(bought, (di, bi), rows["bought"].to_numpy(dtype=float))
    np.add.at(sold, (di, bi), rows["sold"].to_numpy(dtype=float))
    np.add.at(trades, di, rows["buy_trades"].to_numpy(dtype=float))

    wb, ws = _rolling_sum(bought, window), _rolling_sum(sold, window)
    total_b, total_s = wb.sum(axis=1), ws.sum(axis=1)
    net = np.clip(wb - ws, 0, None)
    k = min(5, net.shape[1])
    top = np.partition(net, net.shape[1] - k, axis=1)[:, -k:].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        top5 = np.where(total_b > 0, top / total_b, np.nan)
        bhhi = np.where(total_b > 0, (wb ** 2).sum(axis=1) / total_b ** 2, np.nan)
        shhi = np.where(total_s > 0, (ws ** 2).sum(axis=1) / total_s ** 2, np.nan)
    return pd.DataFrame({"date": dates, "top5_net_buy": top5, "buyer_hhi": bhhi, "seller_hhi": shhi,
                         "trades": _rolling_sum(trades[:, None], window)[:, 0].astype("int64")})


def broker_features(daily_stats, window=20):
    """Per (date, symbol) rolling broker features over the symbol's last `window` trading days."""
    if daily_stats.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    stats = daily_stats.copy()
    stats["date"] = pd.to_datetime(stats["date"])
    if "buy_trades" not in stats:
        stats["buy_trades"] = 0
    frames = []
    for symbol, rows in stats.groupby("symbol", sort=True):
        feats = _symbol_features(rows, window)
        feats.insert(1, "symbol", symbol)
        frames.append(feats)
    return pd.concat(frames, ignore_index=True)[FEATURE_COLUMNS].sort_values(["date", "symbol"], ignore_index=True)


def build(repo_dir=DEFAULT_REPO_DIR, cache_dir=DEFAULT_CACHE_DIR, out=DEFAULT_OUT, window=20, sync=True,
          log=print):
    """Sync, process new days, rebuild the features file; returns (features, processed, malformed)."""
    if sync:
        sync_floorsheet(repo_dir)
    processed, malformed = process_floorsheets(Path(repo_dir) / FLOORSHEET_FOLDER, cache_dir, log=log)
    log(f"processed {len(processed)} new day(s); {len(malformed)} malformed")
    feats = broker_features(load_daily_stats(cache_dir), window=window)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False, date_format="%Y-%m-%d", float_format="%.6g", compression="gzip")
    log(f"wrote {len(feats)} rows to {out}")
    return feats, processed, malformed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--no-sync", action="store_true", help="Do not clone/update the floorsheet first")
    args = parser.parse_args(argv)
    build(args.repo_dir, args.cache_dir, args.out, args.window, sync=not args.no_sync)
    return 0


if __name__ == "__main__":
    sys.exit(main())
