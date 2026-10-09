"""Forecast the next trading days for one NEPSE symbol.

Usage:
    python -m nepse_kronos.predict --symbol NABIL --pred-len 10
    python -m nepse_kronos.predict --symbol NEPSE_INDEX --model finetuned/NEPSE_daily/basemodel/best_model \
        --tokenizer finetuned/NEPSE_daily/tokenizer/best_model
"""
import argparse
from pathlib import Path

import pandas as pd

from nepse_kronos.forecast import forecast_next, load_predictor, plot_forecast
from nepse_kronos.trading_calendar import load_holidays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--clean-dir", default="data/nepse/clean")
    parser.add_argument("--out-dir", default="outputs/nepse")
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--model", default="NeoQuasar/Kronos-small")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda:0 (auto-detected if omitted)")
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    df = pd.read_csv(Path(args.clean_dir) / f"{symbol}.csv", parse_dates=["timestamps"])
    predictor = load_predictor(args.model, args.tokenizer, device=args.device)
    pred = forecast_next(predictor, df, args.pred_len, args.lookback, load_holidays(), args.sample_count)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / f"pred_{symbol}.csv", date_format="%Y-%m-%d")
    plot_forecast(df, pred, out_dir / f"pred_{symbol}.png", f"{symbol} - Kronos forecast")
    print(pred.round(2).to_string())
    print(f"Saved {out_dir / f'pred_{symbol}.csv'} and .png")


if __name__ == "__main__":
    main()
