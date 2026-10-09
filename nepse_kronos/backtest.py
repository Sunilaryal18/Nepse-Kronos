"""Walk-forward backtest of Kronos forecasts on one NEPSE symbol.

Usage:
    python -m nepse_kronos.backtest --symbol NEPSE_INDEX --pred-len 5 --step 10
    python -m nepse_kronos.backtest --symbol NEPSE_INDEX --start-date 2025-06-01 \
        --model finetuned/NEPSE_daily/basemodel/best_model --tokenizer finetuned/NEPSE_daily/tokenizer/best_model
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nepse_kronos.forecast import FEATURES, load_predictor


def walk_forward(predictor, df, lookback, pred_len, step, start_date=None, sample_count=50):
    """Forecast pred_len days from every step-th origin and compare with what really happened."""
    rows = []
    start = pd.Timestamp(start_date) if start_date is not None else None
    for end in range(lookback, len(df) - pred_len + 1, step):
        history = df.iloc[end - lookback:end].reset_index(drop=True)
        actual = df.iloc[end:end + pred_len].reset_index(drop=True)
        origin = history["timestamps"].iloc[-1]
        if start is not None and origin < start:
            continue
        pred = predictor.predict(
            df=history[FEATURES],
            x_timestamp=history["timestamps"],
            y_timestamp=actual["timestamps"],
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=sample_count,
            verbose=False,
        )
        last_close = history["close"].iloc[-1]
        pred_close = float(pred["close"].iloc[-1])
        actual_close = actual["close"].iloc[-1]
        rows.append({
            "origin": origin,
            "target": actual["timestamps"].iloc[-1],
            "last_close": last_close,
            "pred_close": pred_close,
            "actual_close": actual_close,
            "pred_return": pred_close / last_close - 1.0,
            "actual_return": actual_close / last_close - 1.0,
        })
    return pd.DataFrame(rows)


def summarize(results):
    actual = results["actual_close"]
    return {
        "windows": len(results),
        "direction_accuracy": float((np.sign(results["pred_return"]) == np.sign(results["actual_return"])).mean()),
        "mape": float(((results["pred_close"] - actual).abs() / actual).mean()),
        "naive_mape": float(((results["last_close"] - actual).abs() / actual).mean()),
        "rank_ic": float(results["pred_return"].rank().corr(results["actual_return"].rank())),  # Spearman without scipy
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--clean-dir", default="data/nepse/clean")
    parser.add_argument("--out-dir", default="outputs/nepse")
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--start-date", default=None, help="Only evaluate origins on/after this date")
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--model", default="NeoQuasar/Kronos-small")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    df = pd.read_csv(Path(args.clean_dir) / f"{symbol}.csv", parse_dates=["timestamps"])
    predictor = load_predictor(args.model, args.tokenizer, device=args.device)
    results = walk_forward(predictor, df, args.lookback, args.pred_len, args.step, args.start_date, args.sample_count)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / f"backtest_{symbol}.csv", index=False, date_format="%Y-%m-%d")
    for key, value in summarize(results).items():
        print(f"{key:>20}: {value:.4f}" if isinstance(value, float) else f"{key:>20}: {value}")


if __name__ == "__main__":
    main()
