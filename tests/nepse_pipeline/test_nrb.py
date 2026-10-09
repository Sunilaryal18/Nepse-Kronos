from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from nepse_kronos import nrb

FIX = Path(__file__).parent / "fixtures" / "nrb"


@pytest.fixture(scope="module")
def tables_wb():
    return openpyxl.load_workbook(FIX / "cmefs_tables_small.xlsx", read_only=True, data_only=True)


@pytest.fixture(scope="module")
def bfs_wb():
    return openpyxl.load_workbook(FIX / "bfs_small.xlsx", read_only=True, data_only=True)


def ts(s):
    return pd.Timestamp(s)


def test_fiscal_month_date():
    assert nrb.fiscal_month_date("2016/17", "August") == ts("2016-08-16")
    assert nrb.fiscal_month_date("2016/17", "December") == ts("2016-12-16")
    assert nrb.fiscal_month_date("2016/17", "January") == ts("2017-01-16")
    assert nrb.fiscal_month_date("2025/26", "July") == ts("2026-07-16")
    assert nrb.fiscal_month_date("2026/27P", " august ") == ts("2026-08-16")


@pytest.mark.parametrize("bs_year,bs_month,expected", [
    (2083, "Shrawan", "2026-08-16"),
    (2083, "Asar", "2026-07-16"),
    (2083, "Baisakh", "2026-05-16"),
    (2082, "Chaitra", "2026-04-16"),
    (2082, "Magh", "2026-02-16"),
    (2082, "Poush", "2026-01-16"),
    (2082, "Mangsir", "2025-12-16"),
    (2073, 4, "2016-08-16"),
])
def test_bs_month_to_reference(bs_year, bs_month, expected):
    assert nrb.bs_month_to_reference(bs_year, bs_month) == ts(expected)


def test_interest_rates(tables_wb):
    df = nrb.parse_interest_rates(tables_wb)
    assert list(df.columns) == ["interbank", "tbill91", "deposit_rate", "lending_rate", "base_rate"]
    assert len(df) == 5
    aug16 = df.loc[ts("2016-08-16")]
    assert aug16.tolist() == pytest.approx([0.82, 0.4399, 3.29, 8.88, 6.1])
    assert df.loc[ts("2026-07-16"), "tbill91"] == pytest.approx(2.32)
    aug26 = df.loc[ts("2026-08-16")]
    assert np.isnan(aug26["tbill91"])  # "-" placeholder
    assert aug26["interbank"] == pytest.approx(2.75)


def test_remittance_decumulated_across_fiscal_year(tables_wb):
    cum = nrb.parse_remittance_cumulative(tables_wb)["remittance_cum"]
    assert cum.loc[ts("2026-07-16")] == pytest.approx(2363129.892, abs=0.01)
    monthly = nrb.decumulate_fiscal(cum)
    assert np.isnan(monthly.loc[ts("2026-05-16")])  # April (its predecessor) not in the fixture
    assert monthly.loc[ts("2026-06-16")] == pytest.approx(2120795.320 - 1916900.990, abs=0.01)
    assert monthly.loc[ts("2026-07-16")] == pytest.approx(2363129.892 - 2120795.320, abs=0.01)
    # August starts a new fiscal year: no subtraction of July's cumulative value.
    assert monthly.loc[ts("2026-08-16")] == pytest.approx(215046.942, abs=0.01)
    assert monthly.loc[ts("2026-09-16")] == pytest.approx(430000.0 - 215046.942, abs=0.01)


def test_decumulate_january_uses_december():
    idx = [ts("2016-12-16"), ts("2017-01-16")]
    out = nrb.decumulate_fiscal(pd.Series([100.0, 130.0], index=idx))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 30.0


def test_cd_ratio_stops_at_first_table(bfs_wb):
    df = nrb.parse_cd_ratio(bfs_wb)
    assert len(df) == 5
    assert df.loc[ts("2016-08-16"), "cd_ratio"] == pytest.approx(80.06)
    assert df.loc[ts("2026-07-16"), "cd_ratio"] == pytest.approx(71.68, abs=0.005)
    assert df.loc[ts("2026-08-16"), "cd_ratio"] == pytest.approx(71.78, abs=0.005)


def test_build_macro(tables_wb, bfs_wb):
    df = nrb.build_macro(tables_wb, bfs_wb)
    assert list(df.columns) == nrb.COLUMNS
    assert df["month"].is_monotonic_increasing and df["month"].is_unique
    assert (df["available_from"] - df["month"] == pd.Timedelta(days=40)).all()
    assert df["month"].min() == ts("2016-08-16") and df["month"].max() == ts("2026-09-16")
    row = df.set_index("month").loc[ts("2026-08-16")]
    assert np.isnan(row["tbill91"])
    assert row["cd_ratio"] == pytest.approx(71.78, abs=0.005)
    assert row["remittance"] == pytest.approx(215046.942, abs=0.01)
    assert row["available_from"] == ts("2026-09-25")
    sep = df.set_index("month").loc[ts("2026-09-16")]  # only remittance known
    assert np.isnan(sep["interbank"]) and np.isnan(sep["cd_ratio"])


def test_find_tables_post():
    title, link = nrb.find_tables_post((FIX / "cmes_feed.xml").read_text())
    assert "one month Data of 2026/27" in title
    assert link == ("https://www.nrb.org.np/red/current-macroeconomic-and-financial-situation-"
                    "tables-based-on-one-month-data-of-2026-27/")


def test_newest_bfs_xlsx():
    ref, url = nrb.newest_bfs_xlsx((FIX / "monthly_statistics.html").read_text())
    assert ref == ts("2026-08-16")
    assert url.endswith("/2026/09/Shrawan_2083_Publish.xlsx")


def test_main_offline(tmp_path):
    cache = tmp_path / "nrb"
    (cache / "tables").mkdir(parents=True)
    (cache / "bfs").mkdir()
    (cache / "tables" / "t.xlsx").write_bytes((FIX / "cmefs_tables_small.xlsx").read_bytes())
    html = (FIX / "monthly_statistics.html").read_text()
    (cache / "monthly_statistics.html").write_text(html)
    (cache / "bfs" / "Shrawan_2083_Publish.xlsx").write_bytes((FIX / "bfs_small.xlsx").read_bytes())
    out = tmp_path / "macro_monthly.csv"
    nrb.main(["--offline", "--cache-dir", str(cache), "--out", str(out)])
    df = pd.read_csv(out)
    assert list(df.columns) == nrb.COLUMNS
    assert df.loc[0, "month"] == "2016-08-16" and df.loc[0, "available_from"] == "2016-09-25"
