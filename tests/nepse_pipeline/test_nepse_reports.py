import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from nepse_kronos import nepse_reports as nr

FIXTURES = Path(__file__).parent / "fixtures" / "nepse_api"


@pytest.fixture
def nabil_items():
    return json.loads((FIXTURES / "reports_131_NABIL.json").read_text())


@pytest.fixture
def company_list():
    return json.loads((FIXTURES / "companies.json").read_text())


def _item(fy="2082-2083", quarter="Fourth Quarter", qid=4, report="Quarterly Report",
          eps=10.0, submitted=("2026-08-06",), app_id=1):
    return {
        "id": app_id,
        "fiscalReport": {
            "quarterMaster": None if quarter is None else {"id": qid, "quarterName": quarter},
            "reportTypeMaster": {"id": 2, "reportName": report},
            "financialYear": {"fyName": "2025-2026", "fyNameNepali": fy},
            "peValue": "18.86", "epsValue": eps, "paidUpCapital": "1,000.5",
            "profitAmount": None, "netWorthPerShare": 247.28,
        },
        "applicationDocumentDetailsList": [{"submittedDate": d} for d in submitted],
    }


def test_parse_real_nabil_response(nabil_items):
    df = nr.parse_reports(nabil_items, "NABIL")
    assert list(df.columns) == nr.COLUMNS
    # annual reports are dropped; only quarterly rows remain
    assert len(df) == 4
    assert set(df["quarter"]) <= {1, 2, 3, 4}
    q4 = df[(df.fiscal_year == "2082-83") & (df.quarter == 4)].iloc[0]
    assert q4.eps == pytest.approx(28.36)
    assert q4.pe == pytest.approx(18.86)
    assert q4.net_worth_per_share == pytest.approx(247.28)
    assert q4.paid_up_capital == pytest.approx(27056996729.0)
    assert q4.submitted == "2026-08-06"
    q3 = df[(df.fiscal_year == "2082-83") & (df.quarter == 3)].iloc[0]
    assert q3.eps == pytest.approx(33.02) and q3.submitted == "2026-04-28"
    old = df[(df.fiscal_year == "2078-79") & (df.quarter == 4)].iloc[0]
    assert (old.eps, old.pe, old.net_worth_per_share, old.submitted) == (26.88, 32.8, 230.37, "2022-08-24")


def test_string_numbers_and_nulls():
    row = nr.parse_item(_item(eps="1,234.5"), "X")
    assert row["eps"] == 1234.5
    assert row["pe"] == 18.86
    assert row["paid_up_capital"] == 1000.5
    assert row["profit_amount"] is None


@pytest.mark.parametrize("eps", [None, "", "null", "abc"])
def test_missing_eps_is_skipped(eps):
    assert nr.parse_item(_item(eps=eps), "X") is None


def test_non_quarterly_and_incomplete_items_are_skipped():
    assert nr.parse_item(_item(report="Annual Report", quarter=None), "X") is None
    assert nr.parse_item(_item(submitted=()), "X") is None
    assert nr.parse_item({}, "X") is None


@pytest.mark.parametrize("name,qid,expected", [
    ("First Quarter", 1, 1), ("Second Quarter", 2, 2), ("Third Quarter", 3, 3),
    ("Fourth Quarter", 4, 4), ("", 3, 3), ("Odd", 9, None),
])
def test_quarter_mapping(name, qid, expected):
    assert nr.quarter_number({"id": qid, "quarterName": name}) == expected


def test_fiscal_year_label():
    assert nr.fiscal_year_label({"fyNameNepali": "2082-2083"}) == "2082-83"
    assert nr.fiscal_year_label({"fyNameNepali": "2078/2079"}) == "2078-79"
    assert nr.fiscal_year_label({}) is None


def test_multi_document_filing_dated_by_first_document():
    row = nr.parse_item(_item(submitted=("2026-08-10", "2026-08-06")), "X")
    assert row["submitted"] == "2026-08-06"


def test_duplicate_quarter_latest_submission_wins():
    items = [
        _item(eps=11.0, submitted=("2026-09-01",), app_id=5),
        _item(eps=10.0, submitted=("2026-08-06",), app_id=9),
        _item(eps=12.0, quarter="Third Quarter", qid=3, submitted=("2026-04-28",), app_id=3),
    ]
    df = nr.parse_reports(items, "X")
    assert len(df) == 2
    assert df[df.quarter == 4].eps.item() == 11.0


def test_equity_companies_filters_instruments(company_list):
    pairs = nr.equity_companies(company_list)
    symbols = [s for _, s in pairs]
    assert (131, "NABIL") in pairs and (139, "NICA") in pairs
    assert "NBB" in symbols  # delisted equities kept
    assert "NCMMF" not in symbols and not any("/" in s for s in symbols)


class FakeClient:
    def __init__(self, responses, fail=()):
        self.responses = responses
        self.fail = set(fail)
        self.calls = []

    def company_list(self):
        raise AssertionError("company list should come from cache")

    def reports(self, company_id):
        self.calls.append(company_id)
        if company_id in self.fail:
            raise RuntimeError("boom")
        return copy.deepcopy(self.responses[company_id])


def test_collect_skips_cached_and_records_failures(tmp_path, nabil_items):
    cached = nr.cache_path(tmp_path, 131, "NABIL")
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps(nabil_items))
    client = FakeClient({139: [_item(eps=20.0)]}, fail={136})
    failed = nr.collect(client, [(131, "NABIL"), (139, "NICA"), (136, "NBB")], tmp_path)
    assert client.calls == [139, 136]
    assert failed == ["NBB"]
    assert nr.cache_path(tmp_path, 139, "NICA").exists()
    assert not nr.cache_path(tmp_path, 136, "NBB").exists()

    # rerun fetches only the one that failed
    client2 = FakeClient({136: []})
    assert nr.collect(client2, [(131, "NABIL"), (139, "NICA"), (136, "NBB")], tmp_path) == []
    assert client2.calls == [136]

    df = nr.build([(131, "NABIL"), (139, "NICA"), (136, "NBB")], tmp_path)
    assert set(df.symbol) == {"NABIL", "NICA"}
    assert df.duplicated(["symbol", "fiscal_year", "quarter"]).sum() == 0


def test_main_offline_writes_csv(tmp_path, nabil_items, company_list):
    cache = tmp_path / "cache"
    (cache).mkdir()
    (cache / "companies.json").write_text(json.dumps(company_list))
    path = nr.cache_path(cache, 131, "NABIL")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(nabil_items))
    out = tmp_path / "fundamentals.csv"
    nr.main(["--offline", "--cache-dir", str(cache), "--output", str(out)])
    df = pd.read_csv(out, dtype={"fiscal_year": str})
    assert list(df.columns) == nr.COLUMNS
    assert len(df) == 4 and set(df.symbol) == {"NABIL"}


def test_all_zero_placeholder_is_skipped():
    item = _item(eps=0.0)
    item["fiscalReport"].update(peValue=0, netWorthPerShare=0, paidUpCapital=0)
    assert nr.parse_item(item, "X") is None


def test_filing_before_quarter_end_is_mislabelled_and_skipped():
    item = _item(fy="2083-2084", submitted=("2026-08-14",))
    item["fiscalReport"]["financialYear"]["fromYear"] = "2026-07-17"
    assert nr.parse_item(item, "X") is None
    ok = _item(fy="2082-2083", submitted=("2026-08-06",))
    ok["fiscalReport"]["financialYear"]["fromYear"] = "2025-07-17"
    assert nr.parse_item(ok, "X")["fiscal_year"] == "2082-83"


def test_collect_refetches_cached_reports_older_than_max_age(tmp_path):
    import json
    import os
    import time

    from nepse_kronos.nepse_reports import cache_path, collect

    class Client:
        def __init__(self):
            self.calls = []

        def reports(self, company_id):
            self.calls.append(company_id)
            return []

    fresh, stale = cache_path(tmp_path, 1, "AAA"), cache_path(tmp_path, 2, "BBB")
    for path in (fresh, stale):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([]))
    old = time.time() - 40 * 86400
    os.utime(stale, (old, old))

    client = Client()
    collect(client, [(1, "AAA"), (2, "BBB")], tmp_path)
    assert client.calls == []                                   # default: never refetch
    collect(client, [(1, "AAA"), (2, "BBB")], tmp_path, max_age_days=30)
    assert client.calls == [2]                                  # only the stale one
