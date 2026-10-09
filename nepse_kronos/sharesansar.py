"""Collect corporate actions, sectors, share issues and promoter lock-ins from ShareSansar
(sectors of delisted/merged companies from MeroLagani).

Politeness: >= 1 s between requests per site, descriptive User-Agent, retries with back-off.
Every raw response is cached (gzip) under data/nepse/source/sharesansar/ and
data/nepse/source/merolagani/, so an interrupted run resumes where it stopped.

Outputs (data/nepse/meta/):
    corporate_actions.csv  symbol, fiscal_year, cash_pct, bonus_pct, book_close, announced
    rights.csv             symbol, ratio_new_per_old, book_close, ratio_raw
    sectors.csv            symbol, name, sector, listed, listing_date
    issues.csv             symbol, issue_type, open_date, close_date, listing_date, units, price, ratio
    lockins.csv            symbol, allot_date, promoter_lock_end, mf_lock_end

Usage:
    python -m nepse_kronos.sharesansar                   # everything (resumes from cache)
    python -m nepse_kronos.sharesansar --refresh-lists   # re-download the list pages (new listings/issues)
    python -m nepse_kronos.sharesansar --symbols NABIL NICA
"""
import argparse
import json
import logging
import math
import re
import threading
from pathlib import Path

import pandas as pd
import requests

from nepse_kronos.http_cache import NotFound, PoliteClient
from nepse_kronos.merolagani import MeroLagani, strip_tags

BASE = "https://www.sharesansar.com"
DEFAULT_CACHE_DIR = "data/nepse/source/sharesansar"
DEFAULT_OUT_DIR = Path("data/nepse/meta")
DEFAULT_CLEAN_DIR = Path("data/nepse/clean")
PAGE = 50  # DataTables page size; the site caps it at 50
log = logging.getLogger("nepse_kronos.sharesansar")

SECTORS = {1: "Commercial Bank", 2: "Corporate Debentures", 3: "Development Bank", 4: "Finance",
           5: "Government Bonds", 6: "Hotel & Tourism", 7: "Hydropower", 8: "Life Insurance",
           9: "Manufacturing", 12: "Microfinance", 13: "Mutual Fund", 15: "Non-Life Insurance",
           16: "Others", 17: "Preference Share", 18: "Promoter Share", 19: "Trading", 20: "Investment"}
NON_EQUITY = {"Corporate Debentures", "Government Bonds", "Mutual Fund", "Preference Share", "Promoter Share"}
ISSUE_TYPES = {1: "ipo", 2: "fpo", 3: "right", 4: "mutual_fund", 5: "ipo_local", 7: "debenture",
               8: "ipo_migrant", 9: "qii"}
AJAX = {"X-Requested-With": "XMLHttpRequest"}

DIVIDEND_COLUMNS = ["symbol", "fiscal_year", "cash_pct", "bonus_pct", "book_close", "announced"]
RIGHTS_COLUMNS = ["symbol", "ratio_new_per_old", "book_close", "ratio_raw"]
SECTOR_COLUMNS = ["symbol", "name", "sector", "listed", "listing_date"]
ISSUE_COLUMNS = ["symbol", "issue_type", "open_date", "close_date", "listing_date", "units", "price", "ratio"]
LOCKIN_COLUMNS = ["symbol", "allot_date", "promoter_lock_end", "mf_lock_end"]


# -- field helpers ---------------------------------------------------------------------------
def clean_date(value):
    """'2023-01-02 [Closed]' -> '2023-01-02'; anything without a YYYY-MM-DD date -> ''."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    return m.group(0) if m else ""


def to_float(value, default=math.nan):
    if value is None:
        return default
    text = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(text) if text else default
    except ValueError:
        return default


def parse_ratio(value):
    """Right-share ratio 'old:new' -> new shares per old share ('100:15' -> 0.15, '4:1' -> 0.25)."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", str(value or ""))
    if not m or float(m.group(1)) == 0:
        return math.nan
    return float(m.group(2)) / float(m.group(1))


def safe_key(symbol):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", symbol).lower()


# -- parsers ---------------------------------------------------------------------------------
def parse_company_page(page):
    """{'company_id', 'token', 'sector', 'name'} from a /company/<symbol> page."""
    cid = re.search(r'id="companyid"[^>]*>\s*(\d+)', page)
    token = re.search(r'name="_token" content="([^"]+)"', page)
    sector = re.search(r"Sector:\s*<span[^>]*>(.*?)</span>", page, re.S)
    name = re.search(r"<td[^>]*>\s*Name\s*</td>\s*<td[^>]*>(.*?)</td>", page, re.S)
    return {"company_id": cid.group(1) if cid else None,
            "token": token.group(1) if token else None,
            "sector": strip_tags(sector.group(1)) if sector else "",
            "name": strip_tags(name.group(1)) if name else ""}


def parse_dividends(symbol, payload):
    rows = []
    for item in payload.get("data", []):
        rows.append({"symbol": symbol,
                     "fiscal_year": (item.get("year") or "").strip(),
                     "cash_pct": to_float(item.get("cash_dividend"), 0.0),
                     "bonus_pct": to_float(item.get("bonus_share"), 0.0),
                     "book_close": clean_date(item.get("bookclose_date")),
                     "announced": clean_date(item.get("announcement_date"))})
    return rows


def parse_rights(symbol, payload):
    rows = []
    for item in payload.get("data", []):
        raw = (item.get("ratio_value") or "").strip()
        rows.append({"symbol": symbol, "ratio_new_per_old": parse_ratio(raw),
                     "book_close": clean_date(item.get("final_date")), "ratio_raw": raw})
    return rows


def parse_company_list(payload, sector):
    return [{"symbol": strip_tags(item.get("symbol")).upper(), "name": strip_tags(item.get("companyname")),
             "sector": sector, "listed": True, "listing_date": clean_date(item.get("listing_date"))}
            for item in payload.get("data", [])]


def parse_issues(payload, issue_type):
    rows = []
    for item in payload.get("data", []):
        company = item.get("company") or {}
        price = to_float(item.get("issue_price"))
        if math.isnan(price):
            price = to_float(item.get("cutoff_price"))
        rows.append({"symbol": strip_tags(company.get("symbol")).upper(), "issue_type": issue_type,
                     "open_date": clean_date(item.get("opening_date")),
                     "close_date": clean_date(item.get("closing_date")),
                     "listing_date": clean_date(item.get("listing_date")),
                     "units": to_float(item.get("total_units")), "price": price,
                     "ratio": parse_ratio(item.get("ratio_value"))})
    return rows


def parse_lockins(payload):
    return [{"symbol": strip_tags(item.get("symbol")).upper(),
             "allot_date": clean_date(item.get("allot_date")),
             "promoter_lock_end": clean_date(item.get("prom_lock_date")),
             "mf_lock_end": clean_date(item.get("mf_lock_date"))}
            for item in payload.get("data", [])]


# -- client ----------------------------------------------------------------------------------
def paginate(fetch):
    """Every payload of a DataTables listing; `fetch(start)` returns the JSON text of one page."""
    payloads, start = [], 0
    while True:
        payload = json.loads(fetch(start))
        payloads.append(payload)
        start += PAGE
        if start >= int(payload.get("recordsTotal") or 0) or not payload.get("data"):
            return payloads


class ShareSansar:
    def __init__(self, client=None, cache_dir=DEFAULT_CACHE_DIR):
        self.client = client or PoliteClient(cache_dir)
        self.token = None
        self._token_lock = threading.Lock()

    def _pages(self, url, key_prefix, params, refresh=False):
        """All rows of a DataTables GET listing, page by page."""
        def fetch(start):
            return self.client.get(url, key=f"{key_prefix}-start-{start}.json", headers=AJAX, refresh=refresh,
                                   params={"draw": 1, "start": start, "length": PAGE, **params})
        return paginate(fetch)

    def company_list(self, refresh=False):
        out = []
        for sid, sector in SECTORS.items():
            for payload in self._pages(f"{BASE}/company-list", f"company-list/sector-{sid}", {"sector": sid},
                                       refresh):
                out.extend(parse_company_list(payload, sector))
        return out

    def issues(self, refresh=False):
        out = []
        for tid, name in ISSUE_TYPES.items():
            for payload in self._pages(f"{BASE}/existing-issues", f"existing-issues/type-{tid}", {"type": tid},
                                       refresh):
                out.extend(parse_issues(payload, name))
        return out

    def lockins(self, refresh=False):
        out = []
        for tid in (1, 0):
            for payload in self._pages(f"{BASE}/promoter-lockin", f"promoter-lockin/type-{tid}",
                                       {"sector": 0, "type": tid}, refresh):
                out.extend(parse_lockins(payload))
        return out

    def company(self, symbol):
        """Parsed company page (None when ShareSansar has no page for the symbol)."""
        key = f"company/{safe_key(symbol)}.html"
        try:
            fresh = self.client.cached(key) is None
            info = parse_company_page(self.client.get(f"{BASE}/company/{symbol.lower()}", key=key))
        except NotFound:
            return None
        if fresh and info["token"]:
            self.token = info["token"]
        return info

    def _refresh_token(self):
        with self._token_lock:
            self.token = parse_company_page(self.client.get(f"{BASE}/company/nabil"))["token"]

    def _post_page(self, endpoint, company_id, start):
        key = f"{endpoint}/{company_id}-start-{start}.json"
        text = self.client.cached(key)
        if text is not None:
            return text
        for attempt in range(2):
            if self.token is None or attempt:
                self._refresh_token()
            try:
                return self.client.post(f"{BASE}/{endpoint}", key=key,
                                        headers={**AJAX, "X-CSRF-Token": self.token},
                                        data={"draw": 1, "start": start, "length": PAGE, "company": company_id})
            except requests.HTTPError as exc:  # 419 = CSRF token expired
                if attempt or exc.response is None or exc.response.status_code not in (401, 403, 419):
                    raise

    def _datatable_post(self, endpoint, company_id):
        return paginate(lambda start: self._post_page(endpoint, company_id, start))

    def actions(self, symbol):
        """(dividend rows, rights rows, company info) for one symbol; None info if no page."""
        info = self.company(symbol)
        if info is None or not info["company_id"]:
            return [], [], info
        dividends = [r for p in self._datatable_post("company-dividend", info["company_id"])
                     for r in parse_dividends(symbol, p)]
        rights = [r for p in self._datatable_post("company-rightshare", info["company_id"])
                  for r in parse_rights(symbol, p)]
        return dividends, rights, info


# -- orchestration ---------------------------------------------------------------------------
def clean_symbols(clean_dir=DEFAULT_CLEAN_DIR):
    return sorted(p.stem for p in Path(clean_dir).glob("*.csv") if not p.stem.endswith("_INDEX"))


def collect_delisted_sectors(ml, symbols, names, results):
    for i, symbol in enumerate(symbols, 1):
        try:
            sector, raw = ml.sector(symbol)
        except Exception as exc:  # keep going; a re-run retries uncached pages
            log.error("merolagani %s: %s", symbol, exc)
            continue
        results[symbol] = {"symbol": symbol, "name": names.get(symbol, ""), "sector": sector, "listed": False,
                           "listing_date": "", "sector_raw": raw}
        if i % 100 == 0:
            log.info("merolagani sectors %d/%d", i, len(symbols))


def collect_actions(ss, symbols, dividends, rights, missing, label):
    for i, symbol in enumerate(symbols, 1):
        try:
            div, rgt, info = ss.actions(symbol)
        except Exception as exc:
            log.error("sharesansar %s: %s", symbol, exc)
            missing.append(symbol)
            continue
        if info is None:
            log.warning("sharesansar has no page for %s", symbol)
            missing.append(symbol)
        dividends.extend(div)
        rights.extend(rgt)
        if i % 50 == 0:
            log.info("%s actions %d/%d", label, i, len(symbols))


def write_csv(rows, columns, path, sort):
    df = pd.DataFrame(rows, columns=columns)
    df = df.drop_duplicates().sort_values(sort, kind="stable").reset_index(drop=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))
    return df


def run(out_dir=DEFAULT_OUT_DIR, clean_dir=DEFAULT_CLEAN_DIR, symbols=None, refresh_lists=False, ss=None, ml=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = ss or ShareSansar()
    ml = ml or MeroLagani()

    listed = ss.company_list(refresh=refresh_lists)
    listed_syms = {r["symbol"] for r in listed}
    log.info("sharesansar lists %d companies", len(listed_syms))
    names = dict(ml.company_list(refresh=refresh_lists))
    delisted = sorted(s for s in names if s not in listed_syms)
    log.info("merolagani lists %d symbols, %d not on sharesansar's current list", len(names), len(delisted))

    targets = symbols or clean_symbols(clean_dir)
    delisted_sectors, dividends, rights, missing = {}, [], [], []
    ml_thread = threading.Thread(target=collect_delisted_sectors, args=(ml, delisted, names, delisted_sectors))
    ml_thread.start()
    collect_actions(ss, targets, dividends, rights, missing, "listed")
    ml_thread.join()

    if not symbols:
        done = set(targets)
        extra = [s for s in delisted if s not in done and delisted_sectors.get(s, {}).get("sector") not in NON_EQUITY]
        log.info("fetching actions for %d delisted/unlisted equity symbols", len(extra))
        collect_actions(ss, extra, dividends, rights, missing, "delisted")

    issues = ss.issues(refresh=refresh_lists)
    lockins = [r for r in ss.lockins(refresh=refresh_lists)
               if r["allot_date"] or r["promoter_lock_end"] or r["mf_lock_end"]]

    sectors = {}
    for row in listed + list(delisted_sectors.values()):
        sectors.setdefault(row["symbol"], row)
    out = {
        "corporate_actions": write_csv(dividends, DIVIDEND_COLUMNS, out_dir / "corporate_actions.csv",
                                       ["symbol", "book_close", "fiscal_year"]),
        "rights": write_csv(rights, RIGHTS_COLUMNS, out_dir / "rights.csv", ["symbol", "book_close"]),
        "sectors": write_csv(list(sectors.values()), SECTOR_COLUMNS, out_dir / "sectors.csv", ["symbol"]),
        "issues": write_csv(issues, ISSUE_COLUMNS, out_dir / "issues.csv", ["symbol", "issue_type", "open_date"]),
        "lockins": write_csv(lockins, LOCKIN_COLUMNS, out_dir / "lockins.csv", ["symbol", "allot_date"]),
    }
    unknown = sorted({r["sector_raw"] for r in delisted_sectors.values()
                      if r["sector"] and r["sector"] == r["sector_raw"] and r["sector"] not in SECTORS.values()})
    if unknown:
        log.warning("unmapped merolagani sector labels: %s", unknown)
    if missing:
        log.warning("%d symbols without sharesansar data: %s", len(missing), " ".join(sorted(missing)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--clean-dir", default=str(DEFAULT_CLEAN_DIR))
    parser.add_argument("--symbols", nargs="+", help="only these symbols' dividends/rights (skips delisted)")
    parser.add_argument("--refresh-lists", action="store_true",
                        help="re-download company lists, issues and lock-ins instead of using the cache")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.out_dir, args.clean_dir, [s.upper() for s in args.symbols] if args.symbols else None,
        args.refresh_lists)


if __name__ == "__main__":
    main()
