import pandas as pd

from nepse_kronos.dashboard import line_chart, render


def _equity():
    days = pd.bdate_range("2026-10-05", periods=4)
    return pd.DataFrame({"portfolio": [1_000_000.0, 998_000.0, 1_002_000.0, 1_007_000.0],
                         "index": [2566.8, 2578.7, 2572.3, 2591.8]}, index=days)


def test_line_chart_indexes_both_series_to_100_and_labels_them():
    svg = line_chart(_equity())
    assert svg.startswith("<svg") and "</svg>" in svg
    assert "Your portfolio" in svg and "NEPSE index" in svg          # direct labels, not colour alone
    assert "data-points=" in svg                                     # hover data for the tooltip


def test_render_contains_every_section():
    state = {"date": "2026-10-08", "holdings": {"NBL": 100}, "cash": 150_000.0, "unsettled": 0.0,
             "sell_orders": {}, "buy_orders": [], "exposure": 0.83, "top_k": 20, "next_refresh_in": 5,
             "start": "2026-10-02", "capital": 1_000_000.0, "features": ["money"]}
    holdings = pd.DataFrame({"symbol": ["NBL"], "sector": ["Commercial Bank"], "units": [100], "bought": [290.0],
                             "last": [297.0]})
    picks = pd.DataFrame({"symbol": ["NTC"], "sector": ["Others"], "price": [902.0], "buy_low": [895.0],
                          "buy_high": [911.0], "eps_growth": [0.2], "pe": [18.3], "pb": [1.6], "volatility": [0.01]})
    movers = pd.DataFrame({"symbol": ["SIKLES", "SAPIL"], "sector": ["Hydropower", "Manufacturing"],
                           "close": [590.0, 1136.0], "change": [0.034, -0.049]})
    page = render(state, _equity(), holdings, picks, movers, next_day=pd.Timestamp("2026-10-12"))
    for text in ["Your portfolio", "NBL", "+2.4%", "Nothing to do", "Top 5 to consider", "NTC", "Rs 895.0 – 911.0",
                 "SIKLES", "SAPIL", "17 of 20", "not investment advice", "2026-10-12"]:
        assert text in page, text
    assert "▲" in page and "▼" in page                                # gains/losses not shown by colour alone


def test_candle_chart_draws_each_day_with_averages_and_buy_range(ohlcv_factory):
    from nepse_kronos.dashboard import candle_chart
    import numpy as np
    df = ohlcv_factory(120, close=np.linspace(100, 120, 120)).set_index("timestamps")
    svg = candle_chart(df, buy_low=118.0, buy_high=121.0, support=115.0, resistance=121.0)
    assert svg.count('class="candle"') == 60                         # last 60 trading days
    assert "20-day avg" in svg and "50-day avg" in svg and "Buy range" in svg
    assert "Support" in svg and "Resistance" in svg


def test_render_shows_why_and_risks_for_each_pick():
    import pandas as pd
    state = {"date": "2026-10-08", "holdings": {}, "cash": 1_000_000.0, "unsettled": 0.0, "sell_orders": {},
             "buy_orders": [], "exposure": None, "top_k": 20, "next_refresh_in": 5, "start": "2026-10-02",
             "capital": 1_000_000.0, "features": []}
    picks = pd.DataFrame({"symbol": ["NTC"], "sector": ["Others"], "price": [902.0], "buy_low": [895.0],
                          "buy_high": [911.0], "eps_growth": [0.2], "pe": [18.3], "pb": [1.6], "volatility": [0.01]})
    details = {"NTC": {"why": ["Business: profit up."], "risks": ["Government decisions."], "chart": "<svg></svg>"}}
    page = render(state, _equity(), pd.DataFrame(columns=["symbol", "sector", "units", "bought", "last"]), picks,
                  pd.DataFrame(columns=["symbol", "sector", "close", "change"]), next_day=pd.Timestamp("2026-10-12"),
                  details=details, record={"picks": 22, "beat": 0.45, "avg": 0.139, "market": 0.128, "positive": 0.73})
    for text in ["Why", "Business: profit up.", "Risks", "Government decisions.", "22 past picks", "45%"]:
        assert text in page, text


def test_my_portfolio_section_only_when_given():
    import pandas as pd
    state = {"date": "2026-10-09", "holdings": {}, "cash": 1_000_000.0, "unsettled": 0.0, "sell_orders": {},
             "buy_orders": [], "exposure": None, "top_k": 20, "next_refresh_in": 5, "start": "2026-10-02",
             "capital": 1_000_000.0, "features": []}
    empty = lambda cols: pd.DataFrame(columns=cols)
    args = (state, _equity(), empty(["symbol", "sector", "units", "bought", "last"]),
            empty(["symbol", "sector", "price", "buy_low", "buy_high", "eps_growth", "pe", "pb", "volatility"]),
            empty(["symbol", "sector", "close", "change"]), pd.Timestamp("2026-10-12"))
    mine = {"rows": pd.DataFrame({"symbol": ["BHL"], "sector": ["Hydropower"], "units": [131], "buy_price": [250.0],
                                  "last": [223.0], "value": [29213.0], "gain": [-3537.0], "gain_pct": [-0.108],
                                  "day_change": [0.019], "note": [""]}),
            "notes": {"BHL": ["Uptrend: price above its rising 50-day average."]},
            "total": {"cost": 32750.0, "value": 29213.0, "gain": -3537.0, "gain_pct": -0.108, "today_rs": 543.0,
                      "today_pct": 0.019, "take_home": 29000.0, "unpriced": []}}
    with_mine = render(*args, mine=mine)
    for text in ["My real portfolio", "BHL", "-3,537", "Uptrend", "29,000"]:
        assert text in with_mine, text
    assert "My real portfolio" not in render(*args)
