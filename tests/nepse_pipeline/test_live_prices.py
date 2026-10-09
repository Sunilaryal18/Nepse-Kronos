import pandas as pd
import pytest

from nepse_kronos.live_prices import parse_index_history, parse_price_history, update_live

INDEX_JSON = {"content": [
    {"businessDate": "2026-10-06", "closingIndex": 2578.73, "openIndex": 2559.2, "highIndex": 2581.58,
     "lowIndex": 2557.3, "turnoverValue": 3073867081.2, "turnoverVolume": 8112978},
    {"businessDate": "2026-10-05", "closingIndex": 2566.76, "openIndex": 2586.23, "highIndex": 2586.23,
     "lowIndex": 2558.35, "turnoverValue": 316478961.0, "turnoverVolume": 900000},
]}
NABIL_JSON = {"content": [
    {"businessDate": "2026-10-06", "totalTradedQuantity": 53949, "totalTradedValue": 28578508.4,
     "highPrice": 532.0, "lowPrice": 528.0, "closePrice": 531.0},
    {"businessDate": "2026-10-05", "totalTradedQuantity": 40000, "totalTradedValue": 21000000.0,
     "highPrice": 530.0, "lowPrice": 525.0, "closePrice": 529.0},
    {"businessDate": "2026-10-02", "totalTradedQuantity": 60404, "totalTradedValue": 31921134.2,
     "highPrice": 533.0, "lowPrice": 525.0, "closePrice": 528.7},
]}


def test_parse_index_history_sorted_with_canonical_columns():
    df = parse_index_history(INDEX_JSON)
    assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert df["timestamps"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-05", "2026-10-06"]
    assert df["close"].tolist() == [2566.76, 2578.73]
    assert df["open"].iloc[0] == 2586.23


def test_parse_price_history_uses_previous_close_as_open():
    df = parse_price_history(NABIL_JSON)
    assert df["timestamps"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-02", "2026-10-05", "2026-10-06"]
    assert df["open"].tolist()[1:] == [528.7, 529.0]          # no opening price published: previous close
    assert df["open"].iloc[0] == 528.7                        # first row: its own close
    assert df.loc[2, "volume"] == 53949 and df.loc[2, "amount"] == pytest.approx(28578508.4)


class FakeClient:
    def __init__(self):
        self.calls = []

    def security_ids(self):
        return {"NABIL": 131, "ADBL": 397}

    def index_history(self, size):
        self.calls.append(("index", size))
        return INDEX_JSON

    def price_history(self, security_id, start, end):
        self.calls.append(("price", security_id, start, end))
        return NABIL_JSON if security_id == 131 else {"content": []}


def test_update_live_writes_only_days_after_the_community_data(tmp_path):
    client = FakeClient()
    written = update_live(client, ["NABIL", "ADBL", "UNKNOWN"], since=pd.Timestamp("2026-10-02"),
                          live_dir=tmp_path, today=pd.Timestamp("2026-10-06"))
    nabil = pd.read_csv(tmp_path / "NABIL.csv")
    assert nabil["timestamps"].tolist() == ["2026-10-05", "2026-10-06"]
    assert nabil["open"].tolist() == [528.7, 529.0]
    assert pd.read_csv(tmp_path / "NEPSE_INDEX.csv")["timestamps"].tolist() == ["2026-10-05", "2026-10-06"]
    assert written == {"NEPSE_INDEX": 2, "NABIL": 2, "ADBL": 0}
    assert ("price", 131, "2026-09-25", "2026-10-06") in client.calls   # a week back, to get the previous close


def test_update_live_with_nothing_new(tmp_path):
    written = update_live(FakeClient(), ["NABIL"], since=pd.Timestamp("2026-10-06"), live_dir=tmp_path,
                          today=pd.Timestamp("2026-10-06"))
    assert written == {"NEPSE_INDEX": 0, "NABIL": 0}
