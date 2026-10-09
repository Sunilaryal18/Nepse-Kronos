import numpy as np
import pandas as pd
import pytest

from nepse_kronos.picks import daily_picks

N = 200


def _stocks(ohlcv_factory):
    days = pd.bdate_range("2024-01-01", periods=N)
    wiggle = lambda amp: 100 * (1 + amp * (-1) ** np.arange(N))
    closes = {"CALM": wiggle(0.002), "STEADY": wiggle(0.006), "JUMPY": wiggle(0.03), "PRICEY": wiggle(0.001),
              "SHRINK": wiggle(0.001), "SOLD": wiggle(0.001), "BLOCKED": wiggle(0.001)}
    return days, {s: ohlcv_factory(N, close=c).set_index("timestamps") for s, c in closes.items()}


def _reports(days):
    rows = []
    # eps one year ago (same quarter) -> eps now; price ~100
    for symbol, before, now in [("CALM", 8, 10), ("STEADY", 8, 10), ("JUMPY", 8, 10), ("PRICEY", 2, 3),
                                ("SHRINK", 10, 9), ("SOLD", 8, 10), ("BLOCKED", 8, 10)]:
        rows += [{"symbol": symbol, "fiscal_year": "2080-81", "quarter": 1, "eps": before,
                  "net_worth_per_share": 80.0, "submitted": days[10]},
                 {"symbol": symbol, "fiscal_year": "2081-82", "quarter": 1, "eps": now,
                  "net_worth_per_share": 80.0, "submitted": days[150]}]
    return pd.DataFrame(rows)


def test_picks_apply_the_screen_and_rank_calmest_first(ohlcv_factory):
    days, stocks = _stocks(ohlcv_factory)
    broker = pd.DataFrame({"date": days[-1], "symbol": list(stocks),
                           "top5_net_buy": [0.2, 0.2, 0.2, 0.2, 0.2, 0.01, 0.2]})
    picks = daily_picks(stocks, {s: "Bank" for s in stocks}, _reports(days), days[-1], broker=broker,
                        blocklist=lambda day: {"BLOCKED"}, top=5, universe=10, min_history=50)
    # PRICEY: P/E 33 > 25; SHRINK: profit fell; SOLD: brokers clearly not buying; BLOCKED: blocklist
    assert picks["symbol"].tolist() == ["CALM", "STEADY", "JUMPY"]
    calm = picks.iloc[0]
    assert calm["eps_growth"] == pytest.approx(0.25)
    assert calm["pe"] == pytest.approx(calm["price"] / 10)
    assert calm["buy_low"] <= calm["price"] <= calm["buy_high"]
    assert calm["buy_high"] == pytest.approx(round(calm["price"] * 1.01, 1))


def test_only_reports_published_before_the_day_count(ohlcv_factory):
    days, stocks = _stocks(ohlcv_factory)
    picks = daily_picks(stocks, {}, _reports(days), days[100], universe=10, min_history=50)
    assert picks.empty                                          # this year's reports aren't out yet on day 100


def test_top_limits_the_number_of_picks(ohlcv_factory):
    days, stocks = _stocks(ohlcv_factory)
    picks = daily_picks(stocks, {}, _reports(days), days[-1], top=2, universe=10, min_history=50)
    assert len(picks) == 2


def test_technicals_from_candles(ohlcv_factory):
    from nepse_kronos.picks import technicals
    up = ohlcv_factory(260, close=np.linspace(100, 160, 260)).set_index("timestamps")
    t = technicals(up, up.index[-1])
    assert t["trend"] == "up" and t["rsi14"] == 100.0           # every day higher: rising trend, fully "overbought"
    assert t["range52"] == pytest.approx(1.0) and t["support"] < t["resistance"]
    flat = ohlcv_factory(260, close=100 * (1 + 0.01 * (-1) ** np.arange(260))).set_index("timestamps")
    assert technicals(flat, flat.index[-1])["rsi14"] == pytest.approx(50.0, abs=5)
    down = ohlcv_factory(260, close=np.linspace(160, 100, 260)).set_index("timestamps")
    assert technicals(down, down.index[-1])["trend"] == "down"


def test_falling_and_overheated_shares_are_skipped(ohlcv_factory):
    days, stocks = _stocks(ohlcv_factory)
    stocks["FALLING"] = ohlcv_factory(N, close=np.linspace(130, 100, N)).set_index("timestamps")
    hot = np.r_[np.full(N - 20, 100.0), np.linspace(100, 125, 20)]
    stocks["HOT"] = ohlcv_factory(N, close=hot).set_index("timestamps")
    reports = pd.concat([_reports(days), _reports(days).query("symbol == 'CALM'").assign(symbol="FALLING"),
                         _reports(days).query("symbol == 'CALM'").assign(symbol="HOT")])
    reports.loc[reports.symbol == "HOT", "net_worth_per_share"] = 100.0
    picks = daily_picks(stocks, {}, reports, days[-1], top=10, universe=20, min_history=50)
    assert "FALLING" not in set(picks.symbol) and "HOT" not in set(picks.symbol)
    assert {"trend", "rsi14", "support", "resistance"} <= set(picks.columns)


def test_explain_gives_reasons_and_risks():
    from nepse_kronos.picks import explain
    pick = pd.Series({"symbol": "NBL", "sector": "Commercial Bank", "price": 297.0, "eps": 30.6, "eps_before": 25.7,
                      "eps_growth": 0.19, "pe": 9.7, "pb": 1.03, "trend": "up", "rsi14": 58.0, "ret20": 0.04,
                      "range52": 0.7, "vol_ratio": 2.4, "volatility": 0.009, "broker_vs_median": 1.3})
    events = pd.DataFrame({"symbol": ["NBL"], "fiscal_year": ["2081/2082"], "cash_pct": [5.0], "bonus_pct": [0.0],
                           "book_close": [pd.Timestamp("2025-12-01")]})
    why, risks = explain(pick, events, pd.DataFrame(columns=["symbol", "book_close"]), pd.Timestamp("2026-10-09"))
    text = " ".join(why + risks)
    for phrase in ["+19%", "9.7", "1.03", "rising", "RSI 58", "2.4×", "5% cash", "interest rates"]:
        assert phrase in text, phrase


def test_explain_flags_old_dividends_and_one_off_profit_jumps():
    from nepse_kronos.picks import explain
    pick = pd.Series({"symbol": "SADBL", "sector": "Development Bank", "price": 400.0, "eps": 49.2, "eps_before": 14.8,
                      "eps_growth": 2.32, "pe": 8.1, "pb": 2.0, "trend": "sideways", "rsi14": 50.0, "ret20": 0.01,
                      "range52": 0.5, "vol_ratio": 1.0, "volatility": 0.009, "broker_vs_median": float("nan")})
    old = pd.DataFrame({"symbol": ["SADBL"], "fiscal_year": ["2068/2069"], "cash_pct": [11.0], "bonus_pct": [0.0],
                        "book_close": [pd.Timestamp("2013-01-01")]})
    why, risks = explain(pick, old, pd.DataFrame(columns=["symbol", "book_close"]), pd.Timestamp("2026-10-09"))
    assert not any("Dividends" in w for w in why)
    assert any("No dividend recorded since FY 2068/2069" in r for r in risks)
    assert any("one-off" in r for r in risks)
    assert any(w.startswith("Chart: sideways") for w in why)
