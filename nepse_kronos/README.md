# Kronos for NEPSE

Daily forecasts for Nepal Stock Exchange symbols using the Kronos foundation model.

```
nepse-open-data ──► fetch ──► data/nepse/raw ──► prepare ──► data/nepse/clean ──┬──► predict   (forecast + chart)
(public GitHub data)                                                         ├──► backtest  (accuracy vs. "no change")
                                                                             └──► finetune_csv (optional fine-tuning)
```

## 1. Setup

Use Python 3.11. The pinned `pandas==2.2.2` does not support Python 3.13 (it crashes in `pd.to_datetime`).

```bash
uv venv --python 3.11 .venv311
uv pip install --python .venv311/bin/python -r requirements.txt pytest pyyaml
source .venv311/bin/activate
```

Run every command below from the repository root.

## 2. Get data

```bash
python -m nepse_kronos.fetch
```

Downloads or updates [socrateai-official/nepse-open-data](https://github.com/socrateai-official/nepse-open-data) (MIT licence, updated daily) and writes one file per symbol to `data/nepse/raw/`:

- stocks keep their ticker: `NABIL.csv`, `NICA.csv`, ...
- indices get an `_INDEX` suffix: `NEPSE_INDEX.csv`, `BANKING_INDEX.csv`, ...

Stock prices are already adjusted for bonus and rights shares. For indices, the `volume` column is turnover in NPR, not shares.
You can also put your own CSV exports in `data/nepse/raw/`. Common column names such as `Date`, `Open`, `High`, `Low`, `LTP`, `Total Traded Quantity` and `Turnover` are recognised.

Use `--symbols NEPSE_INDEX NABIL` to write only some symbols, and `--no-sync` to skip the download.

## 3. Clean

```bash
python -m nepse_kronos.prepare
```

Writes `data/nepse/clean/<SYMBOL>.csv` in the format Kronos expects. Along the way it:

- fixes column names and numbers written with commas
- drops bad rows and duplicate dates
- drops the early close-only history of indices (before late 2016 they have no real candles), which otherwise makes Kronos forecast fake crashes

## 4. Forecast

```bash
python -m nepse_kronos.predict --symbol NABIL --pred-len 10
```

Writes `outputs/nepse/pred_NABIL.csv` and `.png`. Future dates follow the Monday–Friday NEPSE week, skipping the closures listed in `nepse_kronos/holidays.csv`. Keep that file up to date from NEPSE notices.

Useful options:

- `--lookback 128` uses a shorter context; it scored better than the default 400 on the index
- `--sample-count 50` (default) averages 50 sampled paths; fewer samples make results noticeably noisier
- `--device cpu` if the Apple GPU (`mps`) gives trouble

## 5. Check accuracy

```bash
python -m nepse_kronos.backtest --symbol NABIL --pred-len 5 --step 10 --lookback 128
```

Replays history: from every `--step`-th day it forecasts the next `--pred-len` days and compares with what happened. Add `--start-date YYYY-MM-DD` to score only later dates.

| metric | meaning | useful when |
|---|---|---|
| `mape` | average % error of the final forecast close | below `naive_mape` |
| `naive_mape` | same error for "price stays at the last close" | baseline |
| `direction_accuracy` | share of forecasts that got up/down right | above 0.5 |
| `rank_ic` | rank correlation of predicted vs actual moves | clearly above 0 |

### What we found (5-day horizon, October 2026)

| setup | windows | mape | naive | direction | rank_ic |
|---|---|---|---|---|---|
| pretrained, 2017–2026, lookback 400, 5 samples | 186 | 5.0% | 2.4% | 0.48 | −0.09 |
| pretrained, 2017–2026, lookback 128, 5 samples | 213 | 3.2% | 2.5% | 0.45 | −0.08 |
| pretrained, held-out year, lookback 128, 50 samples | 44 | 1.8% | 1.8% | 0.59 | 0.25 |
| fine-tuned, held-out year, lookback 128, 50 samples | 44 | 2.0% | 1.8% | 0.61 | 0.28 |
| NABIL (stock), pretrained, 2011–2026, lookback 128, 50 samples | 335 | 3.4% | 2.8% | 0.47 | 0.01 |

Kronos does **not** yet beat "no change" on error, for the index or for NABIL. Its positive direction and rank results on the latest year are encouraging but not statistically significant with 44 forecasts. Backtest a symbol before relying on its forecasts.

## 6. Fine-tune (optional)

```bash
python -c "
from huggingface_hub import snapshot_download as get
get('NeoQuasar/Kronos-Tokenizer-base', local_dir='pretrained/Kronos-Tokenizer-base')
get('NeoQuasar/Kronos-small', local_dir='pretrained/Kronos-small')
"
cd finetune_csv && python train_sequential.py --config configs/config_nepse_daily.yaml && cd ..
```

This takes about 35 minutes on a Mac CPU for NEPSE_INDEX. The training script uses CUDA if present, otherwise CPU; the Apple GPU is not used. The last 10% of the series is held out from training. Use the result with:

```bash
python -m nepse_kronos.predict --symbol NEPSE_INDEX \
    --model finetuned/NEPSE_daily/basemodel/best_model --tokenizer finetuned/NEPSE_daily/tokenizer/best_model
```

On the index, fine-tuning did not clearly help: it ranked moves about as well but picked up a downward bias. Compare with `backtest --start-date <held-out start>` before adopting a fine-tuned model.

## Tests

```bash
python -m pytest tests/nepse_pipeline -v
```

The tests use a stub predictor, so they never download model weights.

Forecasts are probabilistic research output, not investment advice.
