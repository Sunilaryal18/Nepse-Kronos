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


def test_main_trims_close_only_history(tmp_path):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "clean"
    raw_dir.mkdir()
    (raw_dir / "IDX_INDEX.csv").write_text(
        "timestamps,open,high,low,close,volume\n"
        "2016-11-24,10,10,10,10,\n"
        "2016-11-28,12,12.5,11.8,12.2,500\n"
    )

    main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])

    assert pd.read_csv(out_dir / "IDX_INDEX.csv")["timestamps"].tolist() == ["2016-11-28"]
