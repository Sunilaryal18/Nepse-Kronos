"""Turn raw NEPSE price files into clean CSVs in the canonical Kronos format.

Usage:
    python -m nepse_kronos.prepare                          # every CSV in data/nepse/raw
    python -m nepse_kronos.prepare --symbols NEPSE_INDEX NABIL
"""
import argparse
from pathlib import Path

import pandas as pd

from nepse_kronos.schema import normalize_ohlcv


def prepare_file(raw_path):
    return normalize_ohlcv(pd.read_csv(raw_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/nepse/raw")
    parser.add_argument("--out-dir", default="data/nepse/clean")
    parser.add_argument("--symbols", nargs="*", help="Only these symbols (file names without .csv)")
    args = parser.parse_args(argv)

    wanted = {s.upper() for s in args.symbols} if args.symbols else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in sorted(Path(args.raw_dir).glob("*.csv")):
        symbol = raw_path.stem.upper()
        if wanted and symbol not in wanted:
            continue
        clean = prepare_file(raw_path)
        clean.to_csv(out_dir / f"{symbol}.csv", index=False, date_format="%Y-%m-%d")
        print(f"{symbol}: {len(clean)} rows, {clean['timestamps'].min().date()} -> {clean['timestamps'].max().date()}")


if __name__ == "__main__":
    main()
