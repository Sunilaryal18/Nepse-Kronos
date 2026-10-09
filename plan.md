# Kronos for NEPSE (Nepal Stock Exchange) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast daily NEPSE stock/index candles with Kronos, evaluate the forecasts with a walk-forward backtest, and optionally fine-tune Kronos on NEPSE data.

**Architecture:** A small, self-contained package `nepse_kronos/` sits next to the existing `model/` package and does not modify it. Data flows in one direction:

```
nepse-open-data ──► nepse_kronos.fetch ──► raw CSV ──► nepse_kronos.prepare ──► clean CSV ──┬──► nepse_kronos.predict   (forecast + chart)
(GitHub, public,    (one file per         (data/      (normalize columns,       (data/nepse/ ├──► nepse_kronos.backtest  (walk-forward metrics)
 bonus/rights-       symbol)               nepse/raw)  validate rows)            clean)      └──► finetune_csv/train_sequential.py (fine-tune)
 adjusted)
```

Future dates are generated on a **Monday–Friday** NEPSE calendar minus a user-maintained holiday list. The Kronos model itself (`model/kronos.py`) is market-agnostic and is used as-is through `KronosPredictor`.

**Tech Stack:** Python 3.11, pandas, numpy, torch, matplotlib, pytest, Hugging Face Hub (`NeoQuasar/Kronos-*` weights), existing `finetune_csv/` pipeline (needs `pyyaml`).

**Spec:** No separate spec document. Requirements come from the conversation:
- Run Kronos on NEPSE data.
- NEPSE trades **Monday to Friday**; public holidays are closed.
- Work proceeds one step at a time.
- Use publicly available data (chosen: [socrateai-official/nepse-open-data](https://github.com/socrateai-official/nepse-open-data), MIT; see Task 3).

## Global Constraints

- Do **not** modify `model/` or `finetune_csv/*.py`. New code lives in `nepse_kronos/`, tests in `tests/nepse_pipeline/`, config in `finetune_csv/configs/`.
- Data frequency is **daily**. Timestamps are dates normalized to midnight (`hour = minute = 0`).
- Canonical clean CSV columns, in this order: `timestamps,open,high,low,close,volume,amount`. `volume` = shares traded, `amount` = turnover in NPR (the public source has no turnover, so it is estimated as volume × mean price).
- Symbol names: stocks keep their NEPSE ticker (`NABIL`); indices get an `_INDEX` suffix (`NEPSE_INDEX`, `BANKING_INDEX`).
- History before 2026-04-10 was traded Sunday–Thursday; it is used as-is (Kronos receives the real weekday). Only *future* dates use the Monday–Friday calendar.
- Trading weekmask: `"Mon Tue Wed Thu Fri"`. Holidays come from `nepse_kronos/holidays.csv` (`date,name`).
- Context length: `lookback <= 512` for `Kronos-small`/`Kronos-base`.
- Unit tests must never download model weights: they use a stub predictor. Real-model runs are manual verification steps.
- Raw/clean market data, downloaded weights and outputs are not committed (`.gitignore`).
- Run every command from the repository root with `.venv311` activated (`source .venv311/bin/activate`) unless a step says otherwise.

## Out of Scope (separate plans later)

- **Our own bonus/rights price adjustment.** The chosen source already publishes adjusted stock prices. If you later switch to an unadjusted source, add an adjustment step between `fetch` and `prepare`.
- Other data sources ([rajeevpaudel/nepse-history](https://github.com/rajeevpaudel/nepse-history), MIT, trade-level from 2015; ShareSansar/MeroLagani exports). Any CSV dropped into `data/nepse/raw/` is accepted by `prepare`.
- Fine-tuning one model on many stocks at once (the `finetune_csv` loader windows over a single CSV; mixing stocks in one file would create windows that span two stocks).
- Web UI integration (`webui/` already accepts any CSV in the canonical format).

## File Structure

| Path | Responsibility |
|---|---|
| `nepse_kronos/__init__.py` | Package marker |
| `nepse_kronos/schema.py` | Turn any raw export (various column names, comma-formatted numbers) into the canonical OHLCV frame |
| `nepse_kronos/trading_calendar.py` | Load holidays; generate the next N NEPSE trading days (Mon–Fri minus holidays) |
| `nepse_kronos/holidays.csv` | User-maintained NEPSE holiday list |
| `nepse_kronos/fetch.py` | Download/update the public dataset; write one raw CSV per symbol |
| `nepse_kronos/prepare.py` | CLI: raw CSVs → clean CSVs |
| `nepse_kronos/forecast.py` | Load Kronos predictor; forecast the next N trading days; plot |
| `nepse_kronos/predict.py` | CLI: clean CSV → forecast CSV + PNG |
| `nepse_kronos/backtest.py` | Walk-forward evaluation + CLI |
| `nepse_kronos/README.md` | How to use the pipeline end to end |
| `finetune_csv/configs/config_nepse_daily.yaml` | Fine-tuning config for one NEPSE series |
| `tests/nepse_pipeline/conftest.py` | Shared fixtures: synthetic OHLCV factory, stub predictor |
| `tests/nepse_pipeline/test_*.py` | Unit tests per module |

---

### Task 0: Environment and branch

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: a `.venv` with Python 3.11 and all dependencies; branch `feature/nepse`

The machine's default `python3` is 3.13, but `requirements.txt` pins `pandas==2.2.2`, which has no Python 3.13 wheels. Use 3.11.

> **Done (2026-10-04):** The pre-existing `.venv` (Python 3.13 + pandas 2.2.2) **segfaults in `pd.to_datetime`** (pandas 2.2.2 does not support 3.13), so it is left untouched and unused. The working environment is `.venv311` (Python 3.11, created with `uv venv --python 3.11 .venv311`, then `uv pip install --python .venv311/bin/python -r requirements.txt pytest pyyaml`). In every later step, run `source .venv311/bin/activate` first (or call `.venv311/bin/python` directly).

- [x] **Step 1: Create a feature branch**

```bash
git checkout -b feature/nepse
```

- [x] **Step 2: Create a Python 3.11 virtual environment**

With `uv` (recommended):
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt pytest pyyaml
```
Without `uv` (needs Python 3.11 installed, e.g. `brew install python@3.11`):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest pyyaml
```

- [x] **Step 3: Verify the Kronos package imports**

Run: `python -c "from model import Kronos, KronosTokenizer, KronosPredictor; print('ok')"`
Expected: `ok`

- [x] **Step 4: Ignore local data, weights and outputs**

Append to `.gitignore`:
```gitignore

# NEPSE pipeline (local data, weights, outputs)
data/nepse/
pretrained/
finetuned/
outputs/
```

- [x] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore local NEPSE data, weights and outputs"
```

---

### Task 1: Normalize raw exports into the canonical schema

**Files:**
- Create: `nepse_kronos/__init__.py`
- Create: `nepse_kronos/schema.py`
- Create: `tests/nepse_pipeline/conftest.py`
- Test: `tests/nepse_pipeline/test_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `nepse_kronos.schema.CANONICAL_COLUMNS: list[str]` = `["timestamps","open","high","low","close","volume","amount"]`
  - `nepse_kronos.schema.PRICE_COLUMNS: list[str]` = `["open","high","low","close"]`
  - `nepse_kronos.schema.normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame` — returns canonical columns, sorted by date, unique dates, no NaN, positive prices.
  - Fixtures `ohlcv_factory(n, start="2024-01-01", close=None) -> pd.DataFrame` and `stub_predictor` (object with `.predict(...)` and `.calls: list[dict]`).

Raw exports from different Nepali sites name columns differently (`Date`, `LTP`, `Total Traded Quantity`, `Turnover`, ...) and often format numbers as `"1,234.50"`. This task maps all of them onto one schema.

- [x] **Step 1: Create the package marker and shared test fixtures**

`nepse_kronos/__init__.py`:
```python
"""Run Kronos forecasts on Nepal Stock Exchange (NEPSE) daily data."""
```

`tests/nepse_pipeline/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n, start="2024-01-01", close=None):
    """Synthetic daily canonical OHLCV frame on Mon-Fri dates."""
    ts = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 100.0 + n - 1, n) if close is None else np.asarray(close, dtype=float)
    return pd.DataFrame({
        "timestamps": ts,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
        "amount": close * 1000.0,
    })


class StubPredictor:
    """Stands in for KronosPredictor: repeats the last input row for every future step."""

    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.calls.append({"df": df, "x_timestamp": x_timestamp, "y_timestamp": y_timestamp,
                           "pred_len": pred_len, **kwargs})
        last = df.iloc[-1].to_numpy()
        return pd.DataFrame([last] * pred_len, columns=df.columns, index=pd.DatetimeIndex(y_timestamp))


@pytest.fixture
def ohlcv_factory():
    return make_ohlcv


@pytest.fixture
def stub_predictor():
    return StubPredictor()
```

- [x] **Step 2: Write the failing tests**

`tests/nepse_pipeline/test_schema.py`:
```python
import pandas as pd
import pytest

from nepse_kronos.schema import CANONICAL_COLUMNS, normalize_ohlcv


def test_maps_site_specific_column_names_and_parses_commas():
    raw = pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-01"],
        "Open": ["1,000.0", "990"],
        "High": ["1,010", "1,000"],
        "Low": ["995", "985"],
        "LTP": ["1,005", "995"],
        "Total Traded Quantity": ["2,000", "1,500"],
        "Turnover": ["2,010,000", "1,492,500"],
    })
    df = normalize_ohlcv(raw)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["timestamps"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert df.loc[1, "open"] == 1000.0
    assert df.loc[1, "close"] == 1005.0
    assert df.loc[1, "volume"] == 2000.0
    assert df.loc[1, "amount"] == 2010000.0


def test_handles_bom_in_first_header():
    raw = pd.DataFrame({"﻿timestamps": ["2024-01-01"], "open": [1], "high": [2], "low": [1], "close": [2]})
    assert normalize_ohlcv(raw)["timestamps"].iloc[0] == pd.Timestamp("2024-01-01")


def test_missing_volume_and_amount_are_filled():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [10], "high": [12], "low": [9], "close": [11]})
    df = normalize_ohlcv(raw)
    assert df.loc[0, "volume"] == 0.0
    assert df.loc[0, "amount"] == 0.0


def test_missing_price_column_raises():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [10], "high": [12], "low": [9]})
    with pytest.raises(ValueError, match="close"):
        normalize_ohlcv(raw)


def test_drops_bad_rows_and_duplicate_dates():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03"],
        "open": [10, "-", 0, 12],
        "high": [11, 11, 11, 13],
        "low": [9, 9, 9, 11],
        "close": [10, 10, 10, 12.5],
    })
    df = normalize_ohlcv(raw)
    # row 2 has a non-numeric open, row 3 has a zero price, 2024-01-03 keeps the last occurrence
    assert df["timestamps"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-03"]
    assert df.loc[1, "close"] == 12.5


def test_repairs_high_low_that_do_not_contain_open_close():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [10], "high": [10.5], "low": [9.8], "close": [11]})
    df = normalize_ohlcv(raw)
    assert df.loc[0, "high"] == 11.0
    assert df.loc[0, "low"] == 9.8
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/nepse_pipeline/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nepse_kronos.schema'`

- [x] **Step 4: Implement `nepse_kronos/schema.py`**

```python
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
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/nepse_pipeline/test_schema.py -v`
Expected: 6 passed

- [x] **Step 6: Commit**

```bash
git add nepse_kronos/__init__.py nepse_kronos/schema.py tests/nepse_pipeline/
git commit -m "feat(nepse): normalize raw NEPSE price exports to canonical OHLCV"
```

---

### Task 2: NEPSE trading calendar (Monday–Friday + holidays)

**Files:**
- Create: `nepse_kronos/trading_calendar.py`
- Create: `nepse_kronos/holidays.csv`
- Test: `tests/nepse_pipeline/test_trading_calendar.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `nepse_kronos.trading_calendar.NEPSE_WEEKMASK: str` = `"Mon Tue Wed Thu Fri"`
  - `nepse_kronos.trading_calendar.DEFAULT_HOLIDAYS_PATH: pathlib.Path`
  - `load_holidays(path=DEFAULT_HOLIDAYS_PATH) -> list[pd.Timestamp]` (empty list if file missing)
  - `next_trading_days(last_date, n, holidays=(), weekmask=NEPSE_WEEKMASK) -> pd.Series` (name `"timestamps"`, length `n`, all strictly after `last_date`)

Kronos receives the weekday, day and month of every future candle, so the future dates must be real NEPSE trading days. Historical rows are kept as they are; this calendar is used only to build future dates.

- [x] **Step 1: Write the failing tests**

`tests/nepse_pipeline/test_trading_calendar.py`:
```python
import pandas as pd

from nepse_kronos.trading_calendar import load_holidays, next_trading_days


def _dates(series):
    return series.dt.strftime("%Y-%m-%d").tolist()


def test_skips_weekend_after_friday():
    # 2025-01-03 is a Friday
    assert _dates(next_trading_days("2025-01-03", 3)) == ["2025-01-06", "2025-01-07", "2025-01-08"]


def test_skips_holidays():
    out = next_trading_days("2025-01-03", 3, holidays=[pd.Timestamp("2025-01-07")])
    assert _dates(out) == ["2025-01-06", "2025-01-08", "2025-01-09"]


def test_result_is_named_series_without_time_of_day():
    out = next_trading_days(pd.Timestamp("2025-01-06 15:00"), 2)
    assert out.name == "timestamps"
    assert _dates(out) == ["2025-01-07", "2025-01-08"]
    assert (out.dt.hour == 0).all()


def test_load_holidays_reads_csv_and_ignores_comments(tmp_path):
    path = tmp_path / "holidays.csv"
    path.write_text("# comment line\ndate,name\n2025-01-07,Test Day\n2025-01-01,New Year\n")
    assert load_holidays(path) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-07")]


def test_load_holidays_missing_file_returns_empty(tmp_path):
    assert load_holidays(tmp_path / "nope.csv") == []
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nepse_pipeline/test_trading_calendar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nepse_kronos.trading_calendar'`

- [x] **Step 3: Implement `nepse_kronos/trading_calendar.py`**

```python
from pathlib import Path

import pandas as pd

NEPSE_WEEKMASK = "Mon Tue Wed Thu Fri"
DEFAULT_HOLIDAYS_PATH = Path(__file__).parent / "holidays.csv"


def load_holidays(path=DEFAULT_HOLIDAYS_PATH):
    """Read market holidays from a CSV with a 'date' column. Missing file means no holidays."""
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path, comment="#")
    return sorted(pd.to_datetime(df["date"]).dt.normalize().tolist())


def next_trading_days(last_date, n, holidays=(), weekmask=NEPSE_WEEKMASK):
    """The n NEPSE trading days strictly after last_date."""
    start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    days = pd.bdate_range(start=start, periods=n, freq="C", weekmask=weekmask, holidays=list(holidays))
    return pd.Series(days, name="timestamps")
```

- [x] **Step 4: Create the holiday list**

`nepse_kronos/holidays.csv`:
```csv
# NEPSE market holidays (weekdays only; weekends are already excluded).
# Add one row per closure from NEPSE's official holiday notices, format YYYY-MM-DD.
date,name
```
Then add the real closure dates for the current and next fiscal year from the NEPSE notices page (nepalstock.com → Notices).

> **Done (2026-10-04):** seeded with 9 closures observed in the price data since 2026-04-10 and 13 upcoming weekday public holidays (Dashain, Tihar, Udhauli, Christmas, Maghe Sankranti, Gyalpo Lhosar) from the official 2083 BS list, converted BS→AD with `nepali-datetime`. Forecasts work without this file; holidays only shift the future dates by a day.

- [x] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/nepse_pipeline/test_trading_calendar.py -v`
Expected: 5 passed

- [x] **Step 6: Commit**

```bash
git add nepse_kronos/trading_calendar.py nepse_kronos/holidays.csv tests/nepse_pipeline/test_trading_calendar.py
git commit -m "feat(nepse): Mon-Fri trading calendar with holiday list"
```

---

### Task 3: Fetch public NEPSE data (socrateai-official/nepse-open-data)

**Files:**
- Create: `nepse_kronos/fetch.py`
- Test: `tests/nepse_pipeline/test_fetch.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `SOURCE_REPO: str`, `DEFAULT_SOURCE_DIR: pathlib.Path` (= `data/nepse/source/nepse-open-data`), `PRICE_FOLDERS: list[str]` (= `["ohlc_index", "ohlc_adjusted_stock"]`)
  - `sync_source(source_dir=DEFAULT_SOURCE_DIR) -> None` — clone (first run) or update the dataset, price folders only
  - `read_daily_files(folder) -> pd.DataFrame` — columns `date, open, high, low, close, volume, symbol`; one row per (symbol, date)
  - `symbol_name(raw_symbol, is_index) -> str` — `"NEPSE_index"` → `"NEPSE_INDEX"`, `"Development%20Bank_index"` → `"DEVELOPMENT_BANK_INDEX"`, `"nabil"` → `"NABIL"`
  - `write_symbol_files(df, out_dir, symbols=None) -> list[str]` — one `<SYMBOL>.csv` per symbol with columns `timestamps,open,high,low,close,volume`
  - CLI `python -m nepse_kronos.fetch` writing `data/nepse/raw/<SYMBOL>.csv`

**Source:** [socrateai-official/nepse-open-data](https://github.com/socrateai-official/nepse-open-data) (MIT licence, updated daily). Checked on 2026-10-04:
- `ohlc_index/adj_YYYY-MM-DD.csv` — one file per trading day, one row per index (17 indices incl. `NEPSE_index`), 2003-07-17 onward. Older files have columns `open,high,low,close,volume,symbol,date`; newer files add `timestamp,date_unix`. Index volume is blank on ~55% of days (mostly older).
- `ohlc_adjusted_stock/adj_YYYY-MM-DD.csv` — columns `date,close,open,high,low,volume,symbol`, ~350 stocks, 2011-07-15 onward. **Prices are already adjusted for bonus/rights shares** (verified on NABIL, NICA, UPPER, NTC: every >15% raw drop on a bonus date is a normal move in the adjusted series), so this plan does no corporate-action adjustment of its own.
- Data-quality issue: two files in the *unadjusted* folder (`unadj_2025-07-07.csv`, `unadj_2025-07-08.csv`) contain unresolved git merge-conflict markers. The reader must survive such lines in any folder: rows whose `date` does not parse are dropped, and for duplicated (symbol, date) rows the last one wins.
- Stock files have no turnover column; `amount` is estimated later by `normalize_ohlcv` as volume × mean price.

- [ ] **Step 1: Write the failing tests**

`tests/nepse_pipeline/test_fetch.py`:
```python
import pandas as pd

from nepse_kronos.fetch import main, read_daily_files, symbol_name, write_symbol_files
from nepse_kronos.schema import normalize_ohlcv


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_read_daily_files_handles_mixed_columns_and_merge_conflicts(tmp_path):
    _write(tmp_path / "ohlc_index" / "adj_2003-07-17.csv",
           "open,high,low,close,volume,symbol,date\n"
           "199.33,199.33,199.33,199.33,,NEPSE_index,2003-07-17\n")
    _write(tmp_path / "ohlc_index" / "adj_2026-10-01.csv",
           "timestamp,open,high,low,close,volume,symbol,date_unix,date\n"
           "1790812800.0,2597.55,2605.64,2589.14,2599.15,4891876000.0,NEPSE_index,1790812800.0,2026-10-01\n"
           "<<<<<<< HEAD\n"
           "1790812800.0,1,1,1,1,1,NEPSE_index,1790812800.0,2026-10-01\n"
           "=======\n"
           "1790812800.0,2597.55,2605.64,2589.14,2599.15,4891876000.0,NEPSE_index,1790812800.0,2026-10-01\n"
           ">>>>>>> 1a2b3c4\n")
    df = read_daily_files(tmp_path / "ohlc_index")

    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "symbol"]
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2003-07-17", "2026-10-01"]
    assert df["close"].tolist() == [199.33, 2599.15]
    assert pd.isna(df["volume"].iloc[0])


def test_symbol_name():
    assert symbol_name("NEPSE_index", is_index=True) == "NEPSE_INDEX"
    assert symbol_name("Development%20Bank_index", is_index=True) == "DEVELOPMENT_BANK_INDEX"
    assert symbol_name("Sen.%20Float_index", is_index=True) == "SEN_FLOAT_INDEX"
    assert symbol_name("nabil", is_index=False) == "NABIL"


def test_write_symbol_files_sorted_per_symbol_and_filtered(tmp_path):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"]),
        "open": [11.0, 10.0, 50.0], "high": [12.0, 11.0, 51.0], "low": [10.0, 9.0, 49.0],
        "close": [11.5, 10.5, 50.5], "volume": [100.0, 200.0, 300.0],
        "symbol": ["AAA", "AAA", "BBB"],
    })
    written = write_symbol_files(df, tmp_path, symbols=["aaa"])

    assert written == ["AAA"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AAA.csv"]
    out = pd.read_csv(tmp_path / "AAA.csv")
    assert list(out.columns) == ["timestamps", "open", "high", "low", "close", "volume"]
    assert out["timestamps"].tolist() == ["2024-01-01", "2024-01-02"]
    # the written file is accepted by the Task 1 normalizer
    assert len(normalize_ohlcv(out)) == 2


def test_main_builds_index_and_stock_files_without_network(tmp_path):
    source, out = tmp_path / "source", tmp_path / "raw"
    _write(source / "ohlc_index" / "adj_2026-10-01.csv",
           "open,high,low,close,volume,symbol,date\n"
           "2597.55,2605.64,2589.14,2599.15,4891876000,NEPSE_index,2026-10-01\n"
           "1494.38,1502.34,1489.64,1499.98,835270533,Banking_index,2026-10-01\n")
    _write(source / "ohlc_adjusted_stock" / "adj_2026-10-01.csv",
           "date,close,open,high,low,volume,symbol\n"
           "2026-10-01,310.0,305.5,310.9,303.3,34881,ADBL\n")

    main(["--source-dir", str(source), "--out-dir", str(out), "--no-sync"])

    assert sorted(p.name for p in out.iterdir()) == ["ADBL.csv", "BANKING_INDEX.csv", "NEPSE_INDEX.csv"]
    assert pd.read_csv(out / "ADBL.csv")["close"].tolist() == [310.0]


def test_main_symbols_filter(tmp_path):
    source, out = tmp_path / "source", tmp_path / "raw"
    _write(source / "ohlc_index" / "adj_2026-10-01.csv",
           "open,high,low,close,volume,symbol,date\n"
           "2597.55,2605.64,2589.14,2599.15,4891876000,NEPSE_index,2026-10-01\n")
    _write(source / "ohlc_adjusted_stock" / "adj_2026-10-01.csv",
           "date,close,open,high,low,volume,symbol\n"
           "2026-10-01,310.0,305.5,310.9,303.3,34881,ADBL\n")

    main(["--source-dir", str(source), "--out-dir", str(out), "--no-sync", "--symbols", "nepse_index"])

    assert sorted(p.name for p in out.iterdir()) == ["NEPSE_INDEX.csv"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nepse_pipeline/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nepse_kronos.fetch'`

- [ ] **Step 3: Implement `nepse_kronos/fetch.py`**

```python
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
PRICE_FOLDERS = ["ohlc_index", "ohlc_adjusted_stock"]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/nepse_pipeline/test_fetch.py -v`
Expected: 5 passed

- [ ] **Step 5: Manual check against the real dataset**

Run: `python -m nepse_kronos.fetch`
Expected: first run clones ~110 MB into `data/nepse/source/nepse-open-data` (later runs only fetch new commits), then prints `Wrote ~370 symbol files to data/nepse/raw (data up to 2026-10-..)`.
Then: `head -3 data/nepse/raw/NEPSE_INDEX.csv && tail -2 data/nepse/raw/NEPSE_INDEX.csv`
Expected: starts in 2003-07, ends on the latest trading day with a close around 2,500–2,700.

- [ ] **Step 6: Commit**

```bash
git add nepse_kronos/fetch.py tests/nepse_pipeline/test_fetch.py
git commit -m "feat(nepse): fetch public NEPSE daily prices per symbol"
```

---

### Task 4: `prepare` CLI — raw CSVs to clean CSVs

**Files:**
- Create: `nepse_kronos/prepare.py`
- Test: `tests/nepse_pipeline/test_prepare.py`

**Interfaces:**
- Consumes: `normalize_ohlcv` (Task 1); raw files from Task 3 (or any CSV export placed in `data/nepse/raw/`)
- Produces:
  - `prepare_file(raw_path) -> pd.DataFrame`
  - `main(argv: list[str] | None = None) -> None`; run as `python -m nepse_kronos.prepare`
  - Output files `data/nepse/clean/<SYMBOL>.csv` with `CANONICAL_COLUMNS`, dates as `YYYY-MM-DD`. The symbol is the raw file name without extension, upper-cased.

- [ ] **Step 1: Write the failing tests**

`tests/nepse_pipeline/test_prepare.py`:
```python
import pandas as pd

from nepse_kronos.prepare import main


def test_main_writes_clean_csv(tmp_path):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "clean"
    raw_dir.mkdir()
    (raw_dir / "abc.csv").write_text(
        "Date,Open,High,Low,LTP,Total Traded Quantity,Turnover\n"
        "2024-01-03,50,51,49,50,1000,50000\n"
        "2024-01-01,100,101,99,100,1000,100000\n"
        "2024-01-02,100,101,99,100,,\n"
    )

    main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])

    clean = pd.read_csv(out_dir / "ABC.csv")
    assert list(clean.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert clean["timestamps"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert clean["close"].tolist() == [100.0, 100.0, 50.0]
    assert clean["volume"].tolist() == [1000.0, 0.0, 1000.0]


def test_symbols_filter(tmp_path):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "clean"
    raw_dir.mkdir()
    row = "date,open,high,low,close\n2024-01-01,10,11,9,10\n"
    (raw_dir / "AAA.csv").write_text(row)
    (raw_dir / "BBB.csv").write_text(row)

    main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir), "--symbols", "bbb"])

    assert sorted(p.name for p in out_dir.iterdir()) == ["BBB.csv"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nepse_pipeline/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nepse_kronos.prepare'`

- [ ] **Step 3: Implement `nepse_kronos/prepare.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/nepse_pipeline/test_prepare.py -v`
Expected: 2 passed

- [ ] **Step 5: Manual check with real data**

Run: `python -m nepse_kronos.prepare --symbols NEPSE_INDEX NABIL`
Expected: two lines such as `NEPSE_INDEX: 5350 rows, 2003-07-17 -> 2026-10-..` and `NABIL: 34xx rows, 2011-07-17 -> 2026-10-..`. Then check the adjusted stock series has no fake crashes:

Run: `python -c "import pandas as pd; c = pd.read_csv('data/nepse/clean/NABIL.csv')['close']; print((c.pct_change() < -0.15).sum())"`
Expected: `0`

- [ ] **Step 6: Commit**

```bash
git add nepse_kronos/prepare.py tests/nepse_pipeline/test_prepare.py
git commit -m "feat(nepse): prepare CLI for raw-to-clean conversion"
```

---

### Task 5: Forecast the next N trading days

**Files:**
- Create: `nepse_kronos/forecast.py`
- Create: `nepse_kronos/predict.py`
- Test: `tests/nepse_pipeline/test_forecast.py`

**Interfaces:**
- Consumes: `next_trading_days`, `load_holidays` (Task 2); clean CSVs (Task 4); `model.Kronos`, `model.KronosTokenizer`, `model.KronosPredictor` (existing)
- Produces:
  - `FEATURES: list[str]` = `["open","high","low","close","volume","amount"]`
  - `load_predictor(model_name="NeoQuasar/Kronos-small", tokenizer_name="NeoQuasar/Kronos-Tokenizer-base", device=None, max_context=512) -> KronosPredictor` (names may also be local directories, e.g. fine-tuned checkpoints)
  - `forecast_next(predictor, df, pred_len, lookback=400, holidays=(), sample_count=10, T=1.0, top_p=0.9) -> pd.DataFrame` (columns `FEATURES`, index = future trading dates named `timestamps`)
  - `plot_forecast(history, forecast, path, title) -> None`
  - CLI `python -m nepse_kronos.predict --symbol NABIL` writing `outputs/nepse/pred_<SYMBOL>.csv` and `.png`

- [ ] **Step 1: Write the failing tests**

`tests/nepse_pipeline/test_forecast.py`:
```python
import pandas as pd
import pytest

from nepse_kronos.forecast import FEATURES, forecast_next, plot_forecast


def test_uses_last_lookback_rows_and_nepse_future_dates(ohlcv_factory, stub_predictor):
    df = ohlcv_factory(50, start="2024-01-01")  # last row is Fri 2024-03-08
    pred = forecast_next(stub_predictor, df, pred_len=3, lookback=20,
                         holidays=[pd.Timestamp("2024-03-12")], sample_count=4)

    call = stub_predictor.calls[0]
    assert len(call["df"]) == 20
    assert list(call["df"].columns) == FEATURES
    assert call["x_timestamp"].iloc[-1] == pd.Timestamp("2024-03-08")
    assert call["df"]["close"].iloc[-1] == df["close"].iloc[-1]
    assert call["sample_count"] == 4
    assert pred.index.strftime("%Y-%m-%d").tolist() == ["2024-03-11", "2024-03-13", "2024-03-14"]
    assert pred.index.name == "timestamps"


def test_rejects_too_little_history(ohlcv_factory, stub_predictor):
    with pytest.raises(ValueError, match="at least 30"):
        forecast_next(stub_predictor, ohlcv_factory(10), pred_len=5, lookback=30)


def test_plot_writes_png(ohlcv_factory, stub_predictor, tmp_path):
    df = ohlcv_factory(40)
    pred = forecast_next(stub_predictor, df, pred_len=5, lookback=30)
    path = tmp_path / "chart.png"
    plot_forecast(df, pred, path, "TEST")
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nepse_pipeline/test_forecast.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nepse_kronos.forecast'`

- [ ] **Step 3: Implement `nepse_kronos/forecast.py`**

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Kronos, KronosPredictor, KronosTokenizer
from nepse_kronos.trading_calendar import next_trading_days

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def load_predictor(model_name="NeoQuasar/Kronos-small", tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                   device=None, max_context=512):
    """Load Kronos from the Hugging Face Hub or a local checkpoint directory."""
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


def forecast_next(predictor, df, pred_len, lookback=400, holidays=(), sample_count=10, T=1.0, top_p=0.9):
    """Forecast the pred_len NEPSE trading days after the last row of df."""
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows of history, got {len(df)}")
    history = df.iloc[-lookback:].reset_index(drop=True)
    future = next_trading_days(history["timestamps"].iloc[-1], pred_len, holidays)
    pred = predictor.predict(
        df=history[FEATURES],
        x_timestamp=history["timestamps"],
        y_timestamp=future,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    pred.index.name = "timestamps"
    return pred


def plot_forecast(history, forecast, path, title):
    """Last 120 days of close/volume plus the forecast."""
    recent = history.tail(120)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(recent["timestamps"], recent["close"], label="History", color="tab:blue")
    ax1.plot(forecast.index, forecast["close"], label="Forecast", color="tab:red")
    ax1.set_ylabel("Close (NPR)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax2.bar(recent["timestamps"], recent["volume"], color="tab:blue")
    ax2.bar(forecast.index, forecast["volume"], color="tab:red")
    ax2.set_ylabel("Volume (shares)")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Implement the CLI `nepse_kronos/predict.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/nepse_pipeline/test_forecast.py -v`
Expected: 3 passed

- [ ] **Step 6: Manual check with the real model**

Run: `python -m nepse_kronos.predict --symbol NEPSE_INDEX --pred-len 10`
Expected: the first run downloads `Kronos-small` and the tokenizer (~100 MB); the table shows 10 rows dated on Mon–Fri only, with no holiday dates; `outputs/nepse/pred_NEPSE_INDEX.png` shows the red forecast continuing from the blue history without a jump in price level. If `mps` errors on Apple Silicon, re-run with `--device cpu`.

- [ ] **Step 7: Commit**

```bash
git add nepse_kronos/forecast.py nepse_kronos/predict.py tests/nepse_pipeline/test_forecast.py
git commit -m "feat(nepse): forecast next NEPSE trading days with Kronos"
```

---

### Task 6: Walk-forward backtest

**Files:**
- Create: `nepse_kronos/backtest.py`
- Test: `tests/nepse_pipeline/test_backtest.py`

**Interfaces:**
- Consumes: `FEATURES`, `load_predictor` (Task 5); clean CSVs (Task 4)
- Produces:
  - `walk_forward(predictor, df, lookback, pred_len, step, start_date=None, sample_count=5) -> pd.DataFrame` with columns `origin, target, last_close, pred_close, actual_close, pred_return, actual_return`
  - `summarize(results) -> dict` with keys `windows, direction_accuracy, mape, naive_mape, rank_ic`
  - CLI `python -m nepse_kronos.backtest --symbol NEPSE` writing `outputs/nepse/backtest_<SYMBOL>.csv`

**Why:** A forecast is only useful if it beats doing nothing. For each window the backtest hides the next `pred_len` real days, forecasts them, and compares the forecast close on the last day with the real one. `naive_mape` is the error of "price stays at the last close"; Kronos should beat it, and `direction_accuracy` should be above 0.5. Using real future dates from the data means holidays are handled automatically.

- [ ] **Step 1: Write the failing tests**

`tests/nepse_pipeline/test_backtest.py`:
```python
import pandas as pd
import pytest

from nepse_kronos.backtest import summarize, walk_forward


def test_windows_cover_history_without_lookahead(ohlcv_factory, stub_predictor):
    df = ohlcv_factory(30)
    results = walk_forward(stub_predictor, df, lookback=10, pred_len=5, step=5)

    # origins end at rows 9, 14, 19, 24 (row 29 has no 5 future rows left)
    assert len(results) == 4
    for call, row in zip(stub_predictor.calls, results.itertuples()):
        assert call["x_timestamp"].iloc[-1] == row.origin
        assert call["y_timestamp"].iloc[0] > row.origin
        assert len(call["df"]) == 10
    assert results["origin"].iloc[0] == df["timestamps"].iloc[9]
    assert results["target"].iloc[0] == df["timestamps"].iloc[14]
    # the stub repeats the last close, so its predicted return is zero
    assert (results["pred_return"] == 0).all()
    assert results["actual_return"].iloc[0] == pytest.approx(df["close"].iloc[14] / df["close"].iloc[9] - 1)


def test_start_date_skips_earlier_origins(ohlcv_factory, stub_predictor):
    df = ohlcv_factory(30)
    results = walk_forward(stub_predictor, df, lookback=10, pred_len=5, step=5,
                           start_date=df["timestamps"].iloc[15])
    assert results["origin"].tolist() == [df["timestamps"].iloc[19], df["timestamps"].iloc[24]]


def test_summarize_metrics():
    results = pd.DataFrame({
        "last_close": [100.0, 100.0, 100.0, 100.0],
        "pred_close": [110.0, 90.0, 105.0, 95.0],
        "actual_close": [120.0, 80.0, 95.0, 100.0],
        "pred_return": [0.10, -0.10, 0.05, -0.05],
        "actual_return": [0.20, -0.20, -0.05, 0.0],
    })
    s = summarize(results)
    assert s["windows"] == 4
    assert s["direction_accuracy"] == pytest.approx(0.5)
    assert s["mape"] == pytest.approx((10 / 120 + 10 / 80 + 10 / 95 + 5 / 100) / 4)
    assert s["naive_mape"] == pytest.approx((20 / 120 + 20 / 80 + 5 / 95 + 0) / 4)
    assert -1.0 <= s["rank_ic"] <= 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nepse_pipeline/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nepse_kronos.backtest'`

- [ ] **Step 3: Implement `nepse_kronos/backtest.py`**

```python
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


def walk_forward(predictor, df, lookback, pred_len, step, start_date=None, sample_count=5):
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
        "rank_ic": float(results["pred_return"].corr(results["actual_return"], method="spearman")),
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
    parser.add_argument("--sample-count", type=int, default=5)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/nepse_pipeline/test_backtest.py -v`
Expected: 3 passed

- [ ] **Step 5: Manual baseline run with the pretrained model**

Run: `python -m nepse_kronos.backtest --symbol NEPSE_INDEX --pred-len 5 --step 10`
Expected: prints `windows`, `direction_accuracy`, `mape`, `naive_mape`, `rank_ic`. On CPU this takes several minutes (one forecast per window). **Write these numbers down** — they are the baseline that fine-tuning (Task 7) must beat. A useful model has `mape < naive_mape` and `direction_accuracy > 0.5`.

- [ ] **Step 6: Commit**

```bash
git add nepse_kronos/backtest.py tests/nepse_pipeline/test_backtest.py
git commit -m "feat(nepse): walk-forward backtest with naive baseline"
```

---

### Task 7: Fine-tune Kronos on a NEPSE series

**Files:**
- Create: `finetune_csv/configs/config_nepse_daily.yaml`

**Interfaces:**
- Consumes: clean CSV `data/nepse/clean/NEPSE_INDEX.csv` (Task 4); existing `finetune_csv/train_sequential.py`
- Produces: checkpoints `finetuned/NEPSE_daily/tokenizer/best_model/` and `finetuned/NEPSE_daily/basemodel/best_model/`, loadable via `--model`/`--tokenizer` in Tasks 5–6

`finetune_csv` splits the CSV by time: first 80% train, next 10% validation, last 10% unused by training. The backtest then runs only on that last 10%, so the fine-tuned model is scored on days it never saw.

`train_sequential.py` reads `config.json` from the pretrained tokenizer **directory**, so weights must be downloaded locally first (a Hugging Face name alone is not enough).

- [ ] **Step 1: Download pretrained weights locally**

```bash
python -c "
from huggingface_hub import snapshot_download as get
get('NeoQuasar/Kronos-Tokenizer-base', local_dir='pretrained/Kronos-Tokenizer-base')
get('NeoQuasar/Kronos-small', local_dir='pretrained/Kronos-small')
"
```
Expected: `pretrained/Kronos-Tokenizer-base/config.json` and `pretrained/Kronos-small/config.json` exist.

- [ ] **Step 2: Create `finetune_csv/configs/config_nepse_daily.yaml`**

Paths are relative to `finetune_csv/`, because training is run from that directory.
```yaml
# Fine-tune Kronos on one daily NEPSE series (index or stock).
# Run from finetune_csv/:  python train_sequential.py --config configs/config_nepse_daily.yaml

data:
  data_path: "../data/nepse/clean/NEPSE_INDEX.csv"
  lookback_window: 256
  predict_window: 10
  max_context: 512
  clip: 5.0
  # time-ordered split; the last 10% is held out for the backtest
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  tokenizer_epochs: 20
  basemodel_epochs: 10
  batch_size: 16
  log_interval: 50
  num_workers: 2
  seed: 42

  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

model_paths:
  pretrained_tokenizer: "../pretrained/Kronos-Tokenizer-base"
  pretrained_predictor: "../pretrained/Kronos-small"

  exp_name: "NEPSE_daily"
  base_path: "../finetuned/"

  base_save_path: ""
  finetuned_tokenizer: ""

  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "kronos_nepse_daily"
  description: "Kronos fine-tuned on NEPSE daily candles"
  use_comet: false

  train_tokenizer: true
  train_basemodel: true

  skip_existing: false

device:
  use_cuda: true   # falls back to CPU when no CUDA GPU is present
  device_id: 0
```

- [ ] **Step 3: Check the series is long enough**

Each training sample needs `lookback_window + predict_window + 1 = 267` rows, and the validation split (10%) must also hold at least one sample, so the CSV needs **≥ 2,670 rows** (~11 years of daily data). For a shorter series lower `lookback_window` to 128 (needs ≥ 1,390 rows).

Run: `python -c "import pandas as pd; print(len(pd.read_csv('data/nepse/clean/NEPSE_INDEX.csv')))"`
Expected: a number ≥ 2670 (or lower the lookback as above).

- [ ] **Step 4: Train**

On a machine with an NVIDIA GPU (or Google Colab with the repo cloned and `data/nepse/clean/`, `pretrained/` copied over):
```bash
cd finetune_csv
python train_sequential.py --config configs/config_nepse_daily.yaml
cd ..
```
Expected: tokenizer then basemodel training logs; validation loss printed each epoch; final checkpoints at `finetuned/NEPSE_daily/tokenizer/best_model/` and `finetuned/NEPSE_daily/basemodel/best_model/`. CPU-only training of a single daily series is possible but slow (hours).

- [ ] **Step 5: Find the start of the held-out period**

Run: `python -c "import pandas as pd; df = pd.read_csv('data/nepse/clean/NEPSE_INDEX.csv'); print(df['timestamps'].iloc[int(len(df) * 0.9)])"`
Expected: a date, e.g. `2025-05-14`. Use it as `HOLDOUT` below.

- [ ] **Step 6: Compare pretrained vs fine-tuned on the held-out period**

```bash
python -m nepse_kronos.backtest --symbol NEPSE_INDEX --pred-len 5 --step 5 --lookback 256 --start-date HOLDOUT
python -m nepse_kronos.backtest --symbol NEPSE_INDEX --pred-len 5 --step 5 --lookback 256 --start-date HOLDOUT \
    --model finetuned/NEPSE_daily/basemodel/best_model \
    --tokenizer finetuned/NEPSE_daily/tokenizer/best_model
```
Expected: two metric tables over the same windows. Keep the fine-tuned model only if its `mape` is lower and `direction_accuracy` higher than the pretrained one. If it is worse, it is over-fitting: halve `basemodel_epochs` and retrain.

- [ ] **Step 7: Commit**

```bash
git add finetune_csv/configs/config_nepse_daily.yaml
git commit -m "feat(nepse): fine-tuning config for daily NEPSE series"
```

---

### Task 8: Usage documentation

**Files:**
- Create: `nepse_kronos/README.md`

**Interfaces:**
- Consumes: all CLIs from Tasks 4–7
- Produces: an end-to-end guide

- [ ] **Step 1: Write `nepse_kronos/README.md`**

````markdown
# Kronos for NEPSE

Daily forecasts for Nepal Stock Exchange symbols using the Kronos foundation model.

## 1. Setup
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt pytest pyyaml
```

## 2. Get data
```bash
python -m nepse_kronos.fetch          # public data from github.com/socrateai-official/nepse-open-data (MIT)
```
Writes one file per symbol to `data/nepse/raw/` (stocks like `NABIL.csv`, indices like `NEPSE_INDEX.csv`). Stock prices are already adjusted for bonus and rights shares. You can also drop your own CSV exports there (e.g. `Date`, `Open`, `High`, `Low`, `LTP`, `Total Traded Quantity`, `Turnover`).

Keep `nepse_kronos/holidays.csv` (weekday market closures) up to date.

## 3. Clean
```bash
python -m nepse_kronos.prepare
```

## 4. Forecast
```bash
python -m nepse_kronos.predict --symbol NABIL --pred-len 10
```
Writes `outputs/nepse/pred_NABIL.csv` and `.png`. Future dates follow the Monday–Friday NEPSE week minus holidays.

## 5. Check accuracy
```bash
python -m nepse_kronos.backtest --symbol NABIL --pred-len 5 --step 10
```
Trust the model only when `mape < naive_mape` and `direction_accuracy > 0.5`.

## 6. Fine-tune (optional, GPU recommended)
See `finetune_csv/configs/config_nepse_daily.yaml`, then:
```bash
cd finetune_csv && python train_sequential.py --config configs/config_nepse_daily.yaml
```
Use the result with `--model finetuned/NEPSE_daily/basemodel/best_model --tokenizer finetuned/NEPSE_daily/tokenizer/best_model`.

## Tests
```bash
python -m pytest tests/nepse_pipeline -v
```

Forecasts are probabilistic research output, not investment advice.
````

- [ ] **Step 2: Run the full NEPSE test suite**

Run: `python -m pytest tests/nepse_pipeline -v`
Expected: 24 passed

- [ ] **Step 3: Commit**

```bash
git add nepse_kronos/README.md
git commit -m "docs(nepse): end-to-end usage guide"
```

---

## Suggested follow-ups (separate plans)

1. **Scheduled refresh** — run `fetch` + `prepare` + `predict` daily after market close.
2. **Batch forecasting** for many symbols with `KronosPredictor.predict_batch` (all series need the same `lookback` and `pred_len`).
3. **Multi-stock fine-tuning** with a dataset class that windows within each symbol separately.
