"""Download public NEPSE daily prices and split them into one CSV per symbol.

Source: https://github.com/socrateai-official/nepse-open-data (MIT licence).
Stock prices in its ohlc_adjusted_stock folder are already adjusted for bonus and rights shares.

Usage:
    python -m nepse_kronos.fetch                          # update source, write every symbol
    python -m nepse_kronos.fetch --symbols NEPSE_INDEX NABIL
    python -m nepse_kronos.fetch --no-sync                # reuse the local copy
"""
import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

SOURCE_REPO = "https://github.com/socrateai-official/nepse-open-data.git"
DEFAULT_SOURCE_DIR = Path("data/nepse/source/nepse-open-data")
PRICE_FOLDERS = ["ohlc_index", "ohlc_adjusted_stock", "ohlc_unadjusted_stock"]  # unadjusted: to spot missed adjustments
COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]


def sync_source(source_dir=DEFAULT_SOURCE_DIR):
    """Clone the dataset (price folders only) or update an existing copy to the latest commit."""
    source_dir = Path(source_dir)
    if (source_dir / ".git").exists():
        subprocess.run(["git", "-C", str(source_dir), "fetch", "--quiet", "--depth", "1", "origin", "main"], check=True)
        subprocess.run(["git", "-C", str(source_dir), "reset", "--quiet", "--hard", "FETCH_HEAD"], check=True)
    else:
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--filter=blob:none", "--sparse",
                        SOURCE_REPO, str(source_dir)], check=True)
    subprocess.run(["git", "-C", str(source_dir), "sparse-checkout", "set", *PRICE_FOLDERS], check=True)


def read_daily_files(folder):
    """Concatenate one-file-per-day CSVs, tolerating varying columns and merge-conflict debris."""
    frames = [pd.read_csv(path, dtype=str, usecols=lambda c: c in COLUMNS)
              for path in sorted(Path(folder).glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "symbol"])
    return df.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)


def symbol_name(raw_symbol, is_index):
    name = unquote(raw_symbol)
    if is_index:
        name = name.removesuffix("_index")
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return f"{name}_INDEX" if is_index else name


def write_symbol_files(df, out_dir, symbols=None):
    """Write <SYMBOL>.csv per symbol; returns the symbols written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.upper() for s in symbols} if symbols else None
    written = []
    for symbol, rows in df.groupby("symbol", sort=True):
        if wanted and symbol not in wanted:
            continue
        rows = rows.sort_values("date").rename(columns={"date": "timestamps"})
        rows[["timestamps", "open", "high", "low", "close", "volume"]].to_csv(
            out_dir / f"{symbol}.csv", index=False, date_format="%Y-%m-%d")
        written.append(symbol)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--out-dir", default="data/nepse/raw")
    parser.add_argument("--symbols", nargs="*", help="Only these symbols, e.g. NEPSE_INDEX NABIL")
    parser.add_argument("--no-sync", action="store_true", help="Do not clone/update the source first")
    args = parser.parse_args(argv)

    if not args.no_sync:
        sync_source(args.source_dir)
    source = Path(args.source_dir)
    indices = read_daily_files(source / "ohlc_index")
    indices["symbol"] = indices["symbol"].map(lambda s: symbol_name(s, is_index=True))
    stocks = read_daily_files(source / "ohlc_adjusted_stock")
    stocks["symbol"] = stocks["symbol"].map(lambda s: symbol_name(s, is_index=False))

    written = write_symbol_files(pd.concat([indices, stocks], ignore_index=True), args.out_dir, args.symbols)
    latest = max(indices["date"].max(), stocks["date"].max()).date()
    print(f"Wrote {len(written)} symbol files to {args.out_dir} (data up to {latest})")


if __name__ == "__main__":
    main()
