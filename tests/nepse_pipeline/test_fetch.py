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
