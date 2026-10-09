import gzip
import json
import math
from pathlib import Path

import pandas as pd
import pytest
import requests

from nepse_kronos import merolagani as ml
from nepse_kronos import sharesansar as ss
from nepse_kronos.http_cache import NotFound, PoliteClient

FIX = Path(__file__).parent / "fixtures" / "sharesansar"


def load(name):
    return json.loads((FIX / name).read_text())


# -- parsers ---------------------------------------------------------------------------------
def test_parse_dividends_handles_closed_suffix_empty_and_null_fields():
    rows = ss.parse_dividends("NABIL", load("dividend_nabil.json"))
    assert len(rows) == 5
    fy7879 = next(r for r in rows if r["fiscal_year"] == "2078/2079")
    assert fy7879 == {"symbol": "NABIL", "fiscal_year": "2078/2079", "cash_pct": 11.5, "bonus_pct": 18.5,
                      "book_close": "2023-01-02", "announced": "2022-12-12"}
    empty_bonus = next(r for r in rows if r["fiscal_year"] == "2080/2081")
    assert empty_bonus["bonus_pct"] == 0.0 and empty_bonus["cash_pct"] == 10.0
    null_announced = next(r for r in rows if r["fiscal_year"] == "2073/2074")
    assert null_announced["announced"] == "" and null_announced["book_close"] == "2017-09-12"
    no_book_close = next(r for r in rows if r["fiscal_year"] == "2069/2070")  # CTBNL: empty cash, no book close
    assert no_book_close["cash_pct"] == 0.0 and no_book_close["bonus_pct"] == 2.0
    assert no_book_close["book_close"] == ""


@pytest.mark.parametrize("raw, expected", [("100:15", 0.15), ("4:1", 0.25), ("1:1", 1.0),
                                           ("1:0.5", 0.5), (" 10 : 3 ", 0.3)])
def test_parse_ratio_is_new_shares_per_old(raw, expected):
    assert ss.parse_ratio(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "abc", "0:5", "1:"])
def test_parse_ratio_bad_input_is_nan(raw):
    assert math.isnan(ss.parse_ratio(raw))


def test_parse_rights_nica():
    rows = ss.parse_rights("NICA", load("rightshare_nica.json"))
    assert rows[0] == {"symbol": "NICA", "ratio_new_per_old": pytest.approx(0.15), "book_close": "2017-03-20",
                       "ratio_raw": "100:15"}
    assert rows[1]["ratio_new_per_old"] == pytest.approx(0.25)


def test_parse_company_list_strips_links():
    payload = load("company_list_sector1.json")
    rows = ss.parse_company_list(payload, "Commercial Bank")
    assert payload["recordsTotal"] > len(rows)  # paged listing
    assert rows[0] == {"symbol": "NMB", "name": "NMB Bank Limited", "sector": "Commercial Bank", "listed": True,
                       "listing_date": "2018-08-22"}


def test_parse_company_page():
    info = ss.parse_company_page((FIX / "company_nica_excerpt.html").read_text())
    assert info == {"company_id": "446", "token": "TESTTOKEN", "sector": "Commercial Bank",
                    "name": "NIC Asia Bank Limited"}


def test_parse_issues():
    ipo = ss.parse_issues(load("existing_issues_type1.json"), "ipo")[0]
    assert ipo["symbol"] == "BENI" and ipo["issue_type"] == "ipo"
    assert (ipo["open_date"], ipo["close_date"], ipo["listing_date"]) == ("2026-09-07", "2026-09-11", "")
    assert ipo["units"] == 863200.0 and ipo["price"] == 100.0 and math.isnan(ipo["ratio"])
    rights = ss.parse_issues(load("existing_issues_type3.json"), "right")
    assert [r["symbol"] for r in rights] == ["UHEWA", "DORDI"]
    assert rights[1]["ratio"] == pytest.approx(0.56621279)


def test_parse_lockins():
    rows = ss.parse_lockins(load("promoter_lockin_type1.json"))
    assert rows[0] == {"symbol": "SABBL", "allot_date": "2026-01-09", "promoter_lock_end": "2029-01-09",
                       "mf_lock_end": "2026-07-08"}


def test_merolagani_sector_and_name():
    page = (FIX / "merolagani_ctbnl_excerpt.html").read_text()
    assert ml.parse_sector(page) == "Commercial Banks"
    assert ml.normalise_sector(ml.parse_sector(page)) == "Commercial Bank"
    assert ml.parse_company_name(page) == "Commerz and Trust Bank Nepal Ltd."
    assert ml.parse_sector("<html>no table</html>") == ""


@pytest.mark.parametrize("raw, expected", [("Hydro Power", "Hydropower"), ("Non Life Insurance", "Non-Life Insurance"),
                                           ("Manufacturing And Processing", "Manufacturing"),
                                           ("Tradings", "Trading"), ("Hotels", "Hotel & Tourism"),
                                           ("Something New", "Something New"), ("", "")])
def test_normalise_sector(raw, expected):
    assert ml.normalise_sector(raw) == expected


def test_parse_autosuggest():
    rows = ml.parse_autosuggest((FIX / "merolagani_autosuggest.json").read_text())
    assert ("CTBNL", "Commerz and Trust Bank Nepal Ltd.") in rows
    assert len(rows) == 4


# -- cache / client --------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(tmp_path, responses):
    sleeps = []
    session = FakeSession(responses)
    client = PoliteClient(tmp_path, session=session, sleep=sleeps.append, clock=lambda: 0.0)
    return client, session, sleeps


def test_cache_second_call_does_not_hit_network(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse("hello")])
    assert client.get("https://x/a", key="a/b.json") == "hello"
    assert client.get("https://x/a", key="a/b.json") == "hello"
    assert len(session.calls) == 1
    assert gzip.open(tmp_path / "a" / "b.json.gz", "rt").read() == "hello"
    assert "User-Agent" in session.headers


def test_cache_remembers_404(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse("gone", 404)])
    for _ in range(2):
        with pytest.raises(NotFound):
            client.get("https://x/missing", key="company/missing.html")
    assert len(session.calls) == 1


def test_retries_with_backoff_then_succeeds(tmp_path):
    client, session, sleeps = make_client(
        tmp_path, [requests.ConnectionError("boom"), FakeResponse("busy", 503), FakeResponse("ok")])
    assert client.get("https://x/a", key="k") == "ok"
    assert len(session.calls) == 3
    assert [s for s in sleeps if s >= 2] == [2.0, 4.0]


def test_throttle_waits_between_requests(tmp_path):
    sleeps = []
    session = FakeSession([FakeResponse("1"), FakeResponse("2")])
    client = PoliteClient(tmp_path, session=session, sleep=sleeps.append, clock=lambda: 100.0, min_interval=1.0)
    client.get("https://x/1")
    client.get("https://x/2")
    assert sleeps == [1.0]


def test_sharesansar_actions_uses_cache_on_rerun(tmp_path):
    page = (FIX / "company_nica_excerpt.html").read_text()
    rights = (FIX / "rightshare_nica.json").read_text()
    divs = (FIX / "dividend_nabil.json").read_text()
    client, session, _ = make_client(tmp_path, [FakeResponse(page), FakeResponse(divs), FakeResponse(rights)])
    first = ss.ShareSansar(client=client).actions("NICA")
    assert len(session.calls) == 3
    method, url, kwargs = session.calls[1]
    assert method == "POST" and url.endswith("/company-dividend")
    assert kwargs["headers"]["X-CSRF-Token"] == "TESTTOKEN" and kwargs["data"]["company"] == "446"
    second = ss.ShareSansar(client=client).actions("NICA")
    assert len(session.calls) == 3
    assert first[:2] == second[:2] and len(first[1]) == 2


def test_run_writes_outputs(tmp_path, monkeypatch):
    class FakeSS:
        def company_list(self, refresh=False):
            return [{"symbol": "NICA", "name": "NIC Asia", "sector": "Commercial Bank", "listed": True,
                     "listing_date": "2013-06-30"}]

        def actions(self, symbol):
            if symbol == "NICA":
                return (ss.parse_dividends(symbol, load("dividend_nabil.json")),
                        ss.parse_rights(symbol, load("rightshare_nica.json")), {"company_id": "446"})
            return [], [], None

        def issues(self, refresh=False):
            return ss.parse_issues(load("existing_issues_type1.json"), "ipo")

        def lockins(self, refresh=False):
            return ss.parse_lockins(load("promoter_lockin_type1.json"))

    class FakeML:
        def company_list(self, refresh=False):
            return [("NICA", "NIC Asia"), ("CTBNL", "Commerz"), ("XYZMF", "Some Fund")]

        def sector(self, symbol):
            return {"CTBNL": ("Commercial Bank", "Commercial Banks"), "XYZMF": ("Mutual Fund", "Mutual Fund")}[symbol]

    clean = tmp_path / "clean"
    clean.mkdir()
    for name in ["NICA", "NEPSE_INDEX"]:
        (clean / f"{name}.csv").write_text("x\n")
    seen = []
    fake_ss = FakeSS()
    orig = fake_ss.actions
    fake_ss.actions = lambda s: (seen.append(s), orig(s))[1]
    out = ss.run(tmp_path / "meta", clean, ss=fake_ss, ml=FakeML())
    assert seen == ["NICA", "CTBNL"]  # index skipped; delisted mutual fund skipped
    sectors = pd.read_csv(tmp_path / "meta" / "sectors.csv")
    assert dict(zip(sectors.symbol, sectors.listed)) == {"CTBNL": False, "NICA": True, "XYZMF": False}
    assert len(out["corporate_actions"]) == 5 and len(out["rights"]) == 2
    for name in ["corporate_actions", "rights", "sectors", "issues", "lockins"]:
        assert (tmp_path / "meta" / f"{name}.csv").exists()
