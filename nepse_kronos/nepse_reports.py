"""Collect quarterly EPS, P/E and net worth per share that companies file with NEPSE.

Source: the NEPSE website API `/api/nots/application/reports/<companyId>`, reached through the
unofficial `nepse` package (github.com/basic-bgnr/NepseUnofficialApi), which produces the
WASM-scrambled access token the site requires. History starts around FY 2078/79 (2022).

Output: data/nepse/meta/fundamentals.csv with one row per (symbol, fiscal_year, quarter):
    symbol, fiscal_year (BS, e.g. "2082-83"), quarter (1-4), eps (annualised, as reported), pe,
    net_worth_per_share, profit_amount, paid_up_capital, submitted (YYYY-MM-DD)
`submitted` is the date the filing appeared on NEPSE; use it as `available_from` (no look-ahead).

Choices:
    * Only "Quarterly Report" filings with a quarter 1-4 and a reported EPS are kept. Annual reports
      have no quarter and repeat audited numbers months later, so they are left out.
    * A filing with several documents is dated by its earliest document.
    * When a company files the same quarter more than once, the latest submission wins.
    * Data-entry errors are dropped: all-zero placeholder filings, and filings dated more than a week
      before their quarter could have ended (e.g. "2083-84 Q4" filed in Aug 2026).
    * Companies come from the NEPSE company list, instrument type "Equity" only (no mutual funds,
      debentures or preference shares); delisted and suspended equities are included.

Raw JSON is cached under data/nepse/source/nepse_api/ (one file per company), so reruns only fetch
what is missing.

Usage:
    python -m nepse_kronos.nepse_reports                 # fetch missing companies, write the CSV
    python -m nepse_kronos.nepse_reports --symbols NABIL NICA
    python -m nepse_kronos.nepse_reports --offline        # rebuild the CSV from the cache only
"""
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/nepse/source/nepse_api")
DEFAULT_OUTPUT = Path("data/nepse/meta/fundamentals.csv")
REPORTS_URL = "/api/nots/application/reports/{company_id}"
COLUMNS = ["symbol", "fiscal_year", "quarter", "eps", "pe", "net_worth_per_share",
           "profit_amount", "paid_up_capital", "submitted"]
EQUITY = "Equity"
QUARTERLY = "quarterly report"
EARLY_TOLERANCE_DAYS = 7  # filings dated earlier than this before the quarter ends are mislabelled
QUARTER_NAMES = {"first quarter": 1, "second quarter": 2, "third quarter": 3, "fourth quarter": 4}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- parsing

def _number(value):
    """Float from a number or numeric string ("1,234.5"); None when missing or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value == value else None
    text = str(value).replace(",", "").strip()
    if not text or text.lower() in {"null", "none", "nan", "-", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fiscal_year_label(fy):
    """Short BS label from the financialYear block: "2082-2083" -> "2082-83"."""
    name = str((fy or {}).get("fyNameNepali") or "").strip()
    parts = [part.strip() for part in name.replace("/", "-").split("-")]
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f"{parts[0]}-{parts[1][-2:]}"
    return name or None


def quarter_number(quarter_master):
    """1-4 from the quarterMaster block (by name, falling back to its id); None otherwise."""
    qm = quarter_master or {}
    name = str(qm.get("quarterName") or "").strip().lower()
    if name in QUARTER_NAMES:
        return QUARTER_NAMES[name]
    qid = _number(qm.get("id"))
    if qid in (1, 2, 3, 4):
        return int(qid)
    return None


def _quarter_end(fy, quarter):
    """Approximate AD end date of a quarter: fiscal-year start (financialYear.fromYear) + 3 months per quarter."""
    start = pd.to_datetime((fy or {}).get("fromYear"), errors="coerce")
    if pd.isna(start):
        return None
    return start + pd.DateOffset(months=3 * quarter) - pd.Timedelta(days=1)


def parse_item(item, symbol):
    """One fundamentals row from a reports item, or None when it is not a usable quarterly report."""
    report = (item or {}).get("fiscalReport") or {}
    report_name = str((report.get("reportTypeMaster") or {}).get("reportName") or "").strip().lower()
    if report_name != QUARTERLY:
        return None
    quarter = quarter_number(report.get("quarterMaster"))
    fiscal_year = fiscal_year_label(report.get("financialYear"))
    eps = _number(report.get("epsValue"))
    dates = sorted(str(doc.get("submittedDate"))[:10]
                   for doc in item.get("applicationDocumentDetailsList") or []
                   if doc.get("submittedDate"))
    if quarter is None or fiscal_year is None or eps is None or not dates:
        return None
    if not any(_number(report.get(k)) for k in ("epsValue", "peValue", "netWorthPerShare", "paidUpCapital")):
        return None  # all-zero placeholder filing
    quarter_end = _quarter_end(report.get("financialYear"), quarter)
    if quarter_end is not None and dates[0] < (quarter_end - pd.Timedelta(days=EARLY_TOLERANCE_DAYS)).strftime("%Y-%m-%d"):
        log.debug("%s %s Q%s filed %s before quarter end: mislabelled, skipped", symbol, fiscal_year, quarter, dates[0])
        return None
    return {
        "symbol": symbol,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "eps": eps,
        "pe": _number(report.get("peValue")),
        "net_worth_per_share": _number(report.get("netWorthPerShare")),
        "profit_amount": _number(report.get("profitAmount")),
        "paid_up_capital": _number(report.get("paidUpCapital")),
        "submitted": dates[0],
        "_application_id": item.get("id") or 0,
    }


def parse_reports(items, symbol):
    """Fundamentals frame for one company; one row per (fiscal_year, quarter), latest submission wins."""
    rows = [row for row in (parse_item(item, symbol) for item in items or []) if row is not None]
    frame = pd.DataFrame(rows, columns=COLUMNS + ["_application_id"])
    numeric = ["eps", "pe", "net_worth_per_share", "profit_amount", "paid_up_capital"]
    frame[numeric] = frame[numeric].astype(float)
    return dedupe(frame)


def dedupe(frame):
    """Keep the latest submission per (symbol, fiscal_year, quarter); ties go to the newer application id."""
    if "_application_id" not in frame.columns:
        frame = frame.assign(_application_id=0)
    frame = frame.sort_values(["symbol", "fiscal_year", "quarter", "submitted", "_application_id"])
    frame = frame.drop_duplicates(["symbol", "fiscal_year", "quarter"], keep="last")
    frame = frame.drop(columns="_application_id").reset_index(drop=True)
    if not frame.empty:
        frame["quarter"] = frame["quarter"].astype(int)
    return frame[COLUMNS]


def equity_companies(company_list):
    """(id, symbol) pairs for equity instruments in the NEPSE company list, sorted by symbol."""
    pairs = {(int(c["id"]), str(c["symbol"]).strip())
             for c in company_list or []
             if c.get("instrumentType") == EQUITY and c.get("symbol") and c.get("id") is not None}
    return sorted(pairs, key=lambda p: (p[1], p[0]))


# ---------------------------------------------------------------- fetching

class NepseClient:
    """Thin wrapper around the `nepse` package with a minimum gap between requests."""

    def __init__(self, min_interval=1.0, retries=3, backoff=5.0, tls_verify=False):
        from nepse import Nepse  # imported lazily so tests and --offline do not need it

        self._nepse = Nepse()
        self._nepse.setTLSVerification(tls_verify)
        self.min_interval = min_interval
        self.retries = retries
        self.backoff = backoff
        self._last = 0.0

    def _wait(self):
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def _call(self, fn, *args):
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                return fn(*args)
            except Exception as exc:  # network, token and recursion errors from the package
                if attempt == self.retries:
                    raise
                log.warning("request failed (%s), retrying in %.0fs", exc, self.backoff * (attempt + 1))
                time.sleep(self.backoff * (attempt + 1))

    def company_list(self):
        return self._call(self._nepse.getCompanyList)

    def reports(self, company_id):
        return self._call(self._nepse.requestGETAPI, REPORTS_URL.format(company_id=company_id))


def cache_path(cache_dir, company_id, symbol):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in symbol)
    return Path(cache_dir) / "reports" / f"{company_id}_{safe}.json"


def load_company_list(client, cache_dir, refresh=False):
    """Company list from the cache, fetched through `client` when missing or `refresh` is set."""
    path = Path(cache_dir) / "companies.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    if client is None:
        raise FileNotFoundError(f"{path} missing and no client to fetch it")
    companies = client.company_list()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(companies))
    return companies


def _needs_fetch(path, max_age_days):
    if not path.exists():
        return True
    return max_age_days is not None and time.time() - path.stat().st_mtime > max_age_days * 86400


def collect(client, companies, cache_dir=DEFAULT_CACHE_DIR, max_age_days=None):
    """Fetch reports for every (id, symbol) not cached yet, or cached more than `max_age_days` ago.

    Returns the symbols that failed.
    """
    failed = []
    todo = [(cid, sym) for cid, sym in companies if _needs_fetch(cache_path(cache_dir, cid, sym), max_age_days)]
    log.info("%d companies, %d cached, %d to fetch", len(companies), len(companies) - len(todo), len(todo))
    for i, (cid, sym) in enumerate(todo, 1):
        try:
            items = client.reports(cid)
        except Exception as exc:
            log.error("%s (id %s) failed: %s", sym, cid, exc)
            failed.append(sym)
            continue
        if not isinstance(items, list):
            log.error("%s (id %s): unexpected response %r", sym, cid, type(items))
            failed.append(sym)
            continue
        path = cache_path(cache_dir, cid, sym)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items))
        tmp.replace(path)
        if i % 25 == 0:
            log.info("fetched %d/%d", i, len(todo))
    return failed


def build(companies, cache_dir=DEFAULT_CACHE_DIR):
    """Fundamentals for all cached companies in `companies`."""
    frames = []
    for cid, sym in companies:
        path = cache_path(cache_dir, cid, sym)
        if path.exists():
            frames.append(parse_reports(json.loads(path.read_text()), sym))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = dedupe(pd.concat(frames, ignore_index=True))
    return out.sort_values(["symbol", "fiscal_year", "quarter"]).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbols", nargs="*", help="limit to these symbols")
    parser.add_argument("--refresh-list", action="store_true", help="re-download the company list")
    parser.add_argument("--offline", action="store_true", help="use cached responses only")
    parser.add_argument("--min-interval", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--max-age-days", type=float, default=None,
                        help="re-download companies whose cached reports are older than this (picks up new quarters)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy

    client = None if args.offline else NepseClient(min_interval=max(args.min_interval, 1.0))
    companies = equity_companies(load_company_list(client, args.cache_dir, refresh=args.refresh_list))
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        companies = [(cid, sym) for cid, sym in companies if sym.upper() in wanted]
    failed = [] if client is None else collect(client, companies, args.cache_dir, args.max_age_days)

    frame = build(companies, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    log.info("wrote %d rows for %d symbols to %s", len(frame), frame["symbol"].nunique(), args.output)
    if failed:
        log.warning("%d companies failed (rerun to retry): %s", len(failed), " ".join(failed))
    return frame


if __name__ == "__main__":
    main()
