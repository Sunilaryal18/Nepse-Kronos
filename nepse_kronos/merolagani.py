"""MeroLagani: full company list (incl. delisted/merged) and each company's sector.

Used for symbols that are no longer listed on ShareSansar's company list. The CLI that writes
the outputs lives in `nepse_kronos.sharesansar`.
"""
import html
import json
import logging
import re

from nepse_kronos.http_cache import NotFound, PoliteClient

BASE = "https://merolagani.com"
DEFAULT_CACHE_DIR = "data/nepse/source/merolagani"
log = logging.getLogger(__name__)

# MeroLagani sector label (lower-cased, letters only) -> ShareSansar sector name.
SECTOR_MAP = {
    "commercialbanks": "Commercial Bank",
    "commercialbank": "Commercial Bank",
    "developmentbanks": "Development Bank",
    "developmentbank": "Development Bank",
    "developmentbanklimited": "Development Bank",
    "finance": "Finance",
    "microfinance": "Microfinance",
    "microcredit": "Microfinance",
    "hydropower": "Hydropower",
    "lifeinsurance": "Life Insurance",
    "nonlifeinsurance": "Non-Life Insurance",
    "hotels": "Hotel & Tourism",
    "hotel": "Hotel & Tourism",
    "hotelsandtourism": "Hotel & Tourism",
    "hotelandtourism": "Hotel & Tourism",
    "manufacturingandprocessing": "Manufacturing",
    "manufacturingandproduction": "Manufacturing",
    "manufacturing": "Manufacturing",
    "investment": "Investment",
    "tradings": "Trading",
    "trading": "Trading",
    "others": "Others",
    "other": "Others",
    "mutualfund": "Mutual Fund",
    "mutualfunds": "Mutual Fund",
    "corporatedebenture": "Corporate Debentures",
    "corporatedebentures": "Corporate Debentures",
    "debenture": "Corporate Debentures",
    "debentures": "Corporate Debentures",
    "promotershare": "Promoter Share",
    "promotershares": "Promoter Share",
    "promotorshare": "Promoter Share",
    "promotorshares": "Promoter Share",
    "governmentbond": "Government Bonds",
    "governmentbonds": "Government Bonds",
    "preferenceshare": "Preference Share",
    "preferenceshares": "Preference Share",
    "preferredstock": "Preference Share",
    "insurance": "Insurance",
}


def strip_tags(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def normalise_sector(raw):
    """Map a MeroLagani sector label onto ShareSansar's naming; unknown labels pass through."""
    raw = re.sub(r"\s+", " ", html.unescape(raw or "")).strip()
    if not raw:
        return ""
    return SECTOR_MAP.get(re.sub(r"[^a-z]", "", raw.lower()), raw)


def parse_autosuggest(text):
    """[(symbol, name)] from AutoSuggestHandler.ashx?type=Company JSON ("l": "SYM (Name)")."""
    out = []
    for item in json.loads(text):
        symbol = (item.get("d") or "").strip().upper()
        if not symbol:
            continue
        label = item.get("l") or ""
        m = re.match(r"^\s*" + re.escape(symbol) + r"\s*\((.*)\)\s*$", label, re.I)
        out.append((symbol, html.unescape(m.group(1) if m else label).strip()))
    return out


def parse_sector(page):
    """Raw sector label from a CompanyDetail.aspx page ('' when absent, e.g. unknown symbol)."""
    m = re.search(r"Sector\s*</th>\s*<td[^>]*>(.*?)</td>", page, re.S | re.I)
    return strip_tags(m.group(1)) if m else ""


def parse_company_name(page):
    m = re.search(r'id="ctl00_ContentPlaceHolder1_CompanyDetail1_companyName"[^>]*>(.*?)</span>', page, re.S)
    if not m:
        return ""
    name = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return re.sub(r"\s*\([A-Z0-9]+\)\s*$", "", name)


class MeroLagani:
    def __init__(self, client=None, cache_dir=DEFAULT_CACHE_DIR):
        self.client = client or PoliteClient(cache_dir)

    def company_list(self, refresh=False):
        text = self.client.get(f"{BASE}/handlers/AutoSuggestHandler.ashx", key="autosuggest/company.json",
                               params={"type": "Company"}, refresh=refresh)
        return parse_autosuggest(text)

    def sector(self, symbol):
        """(normalised sector, raw label) for `symbol`; ('', '') if the page has none."""
        try:
            page = self.client.get(f"{BASE}/CompanyDetail.aspx", key=f"company-detail/{symbol.upper()}.html",
                                   params={"symbol": symbol.upper()})
        except NotFound:
            return "", ""
        raw = parse_sector(page)
        return normalise_sector(raw), raw
