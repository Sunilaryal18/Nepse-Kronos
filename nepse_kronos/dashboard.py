"""One-page NEPSE dashboard: paper portfolio vs the market, holdings, orders, shares to consider, top movers.

Usage:
    python -m nepse_kronos.dashboard             # writes outputs/nepse/dashboard.html (run after paper_trade)
    python -m nepse_kronos.dashboard --open      # ... and opens it in the browser
"""
import argparse
import html
import json
import subprocess
from pathlib import Path

import pandas as pd

from nepse_kronos.benchmark import total_return_index
from nepse_kronos.investor import quality_blocklist, supply_blocklist
from nepse_kronos.costs import CostModel
from nepse_kronos.investor_backtest import load_sectors, tradable_stocks
from nepse_kronos.my_portfolio import HOLDINGS, health_notes, load_holdings, summarize, value_holdings
from nepse_kronos.picks import daily_picks, explain, track_record
from nepse_kronos.trading_calendar import load_holidays, next_trading_days
from nepse_kronos.universe import liquid_universe, load_panel

SERIES = [("portfolio", "Your portfolio", "--series-1"), ("index", "NEPSE index", "--series-2")]

STYLE = """
.viz-root { color-scheme: light; --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --text-muted: #7a7974; --grid: #e6e5e1; --rule: #dcdbd6;
  --series-1: #2a78d6; --series-2: #eb6834; --good: #0a7d0a; --critical: #c23030; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --text-muted: #9a998f; --grid: #2c2c2a; --rule: #3a3a37; --series-1: #3987e5; --series-2: #d95926;
  --good: #3fbf3f; --critical: #e66767; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark; --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f; --grid: #2c2c2a; --rule: #3a3a37;
  --series-1: #3987e5; --series-2: #d95926; --good: #3fbf3f; --critical: #e66767; }
body { margin: 0; background: var(--page); }
.viz-root { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: var(--text-primary);
  background: var(--page); max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 32px 0 8px; }
.sub { color: var(--text-secondary); margin: 0 0 20px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--surface-1); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; }
.tile .label { color: var(--text-secondary); font-size: 12px; } .tile .value { font-size: 22px; font-weight: 600; }
.card { background: var(--surface-1); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--text-secondary); font-weight: 500; font-size: 12px; border-bottom: 1px solid var(--rule); padding: 6px 8px; }
td { padding: 6px 8px; border-bottom: 1px solid var(--grid); } td.num, th.num { text-align: right; white-space: nowrap; }
.up { color: var(--good); } .down { color: var(--critical); }
.note { color: var(--text-secondary); font-size: 13px; margin: 8px 0 0; }
.legend { display: flex; gap: 16px; color: var(--text-secondary); font-size: 12px; margin-bottom: 6px; }
.swatch { display: inline-block; width: 14px; height: 2px; vertical-align: middle; margin-right: 6px; }
.chart { position: relative; } .chart svg { width: 100%; height: auto; display: block; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--rule);
  border-radius: 6px; padding: 6px 8px; font-size: 12px; display: none; white-space: nowrap; }
.pick { margin-top: 12px; } .pick h3 { font-size: 15px; margin: 0 0 6px; }
.pick .facts { color: var(--text-secondary); font-size: 13px; margin: 0 0 8px; }
.pick ul { margin: 6px 0 0; padding-left: 18px; } .pick .label { font-weight: 600; font-size: 13px; margin-top: 8px; }
.candles svg { width: 100%; height: auto; display: block; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
"""

SCRIPT = """
document.querySelectorAll('.chart').forEach(function (box) {
  var svg = box.querySelector('svg'), tip = box.querySelector('.tip'), line = svg.querySelector('.cross');
  var pts = JSON.parse(svg.getAttribute('data-points'));
  svg.addEventListener('mousemove', function (e) {
    var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * svg.viewBox.baseVal.width;
    var best = pts.reduce(function (a, p) { return Math.abs(p.x - x) < Math.abs(a.x - x) ? p : a; });
    line.setAttribute('x1', best.x); line.setAttribute('x2', best.x); line.style.display = 'block';
    tip.innerHTML = '<b>' + best.date + '</b><br>Your portfolio: ' + best.p + '<br>NEPSE index: ' + best.i;
    tip.style.display = 'block';
    tip.style.left = Math.min(r.width - 170, best.x / svg.viewBox.baseVal.width * r.width + 12) + 'px';
    tip.style.top = '8px';
  });
  svg.addEventListener('mouseleave', function () { tip.style.display = 'none'; line.style.display = 'none'; });
});
"""


def _pct(x, digits=1):
    return f"{x:+.{digits}%}"


def _move(x):
    cls, arrow = ("up", "▲") if x > 0 else ("down", "▼") if x < 0 else ("", "•")
    return f'<span class="{cls}">{arrow} {_pct(x)}</span>'


def line_chart(equity, width=720, height=260):
    """Both series indexed to 100 at the start (one axis), direct-labelled, with hover data."""
    pad_l, pad_r, pad_t, pad_b = 44, 120, 14, 28
    scaled = equity / equity.iloc[0] * 100
    low, high = scaled.min().min(), scaled.max().max()
    span = max(high - low, 1.0)
    low, high = low - span * 0.15, high + span * 0.15
    n = len(scaled)
    xs = [pad_l + (width - pad_l - pad_r) * (k / (n - 1) if n > 1 else 0.5) for k in range(n)]
    y = lambda v: pad_t + (height - pad_t - pad_b) * (1 - (v - low) / (high - low))

    parts = []
    for k in range(5):  # recessive grid with value labels
        value = low + (high - low) * k / 4
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(value):.1f}" y2="{y(value):.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>'
                     f'<text x="{pad_l - 6}" y="{y(value) + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="var(--text-muted)">{value:.1f}</text>')
    for k in sorted({0, n // 2, n - 1}):
        parts.append(f'<text x="{xs[k]:.1f}" y="{height - 8}" text-anchor="middle" font-size="11" '
                     f'fill="var(--text-muted)">{scaled.index[k]:%d %b}</text>')
    for column, label, color in SERIES:
        values = scaled[column]
        path = " ".join(f"{'M' if k == 0 else 'L'}{xs[k]:.1f},{y(v):.1f}" for k, v in enumerate(values))
        end_x, end_y = xs[-1], y(values.iloc[-1])
        parts.append(f'<path d="{path}" fill="none" stroke="var({color})" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>'
                     f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4" fill="var({color})" '
                     f'stroke="var(--surface-1)" stroke-width="2"/>'
                     f'<text x="{end_x + 10:.1f}" y="{end_y + 4:.1f}" font-size="12" fill="var(--text-primary)">'
                     f'{label} {values.iloc[-1] - 100:+.1f}%</text>')
    parts.append(f'<line class="cross" y1="{pad_t}" y2="{height - pad_b}" stroke="var(--text-muted)" '
                 f'stroke-width="1" stroke-dasharray="3 3" style="display:none"/>')
    points = [{"x": round(xs[k], 1), "date": f"{scaled.index[k]:%Y-%m-%d}",
               "p": f"{scaled['portfolio'].iloc[k]:.1f}", "i": f"{scaled['index'].iloc[k]:.1f}"} for k in range(n)]
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Portfolio and NEPSE index, both starting at 100" '
            f"data-points='{html.escape(json.dumps(points))}'>{''.join(parts)}</svg>")


def candle_chart(df, buy_low, buy_high, support, resistance, days=60, width=640, height=220):
    """Last `days` candles with 20/50-day averages, support/resistance and the shaded buying range."""
    pad_l, pad_r, pad_t, pad_b = 44, 92, 30, 22
    ma20, ma50 = df["close"].rolling(20).mean().tail(days), df["close"].rolling(50).mean().tail(days)
    tail = df.tail(days)
    values = pd.concat([tail["low"], tail["high"], ma20, ma50, pd.Series([buy_low, buy_high, support, resistance])]).dropna()
    low, high = values.min(), values.max()
    span = max(high - low, 1e-9)
    low, high = low - span * 0.05, high + span * 0.05
    step = (width - pad_l - pad_r) / len(tail)
    x = lambda k: pad_l + step * (k + 0.5)
    y = lambda v: pad_t + (height - pad_t - pad_b) * (1 - (v - low) / (high - low))

    parts = [f'<rect x="{pad_l}" y="{y(buy_high):.1f}" width="{width - pad_l - pad_r}" '
             f'height="{max(y(buy_low) - y(buy_high), 1):.1f}" fill="var(--series-1)" opacity="0.22"/>']
    legend = [("line", "--series-1", "20-day avg"), ("line", "--series-2", "50-day avg"),
              ("band", "--series-1", "Buy range"), ("box", "--good", "Closed higher"), ("box", "--critical", "Closed lower")]
    lx = pad_l
    for kind, color, label in legend:
        fade = ' opacity="0.35"' if kind == "band" else ""
        mark = (f'<line x1="{lx}" x2="{lx + 14}" y1="12" y2="12" stroke="var({color})" stroke-width="2"/>' if kind == "line"
                else f'<rect x="{lx}" y="7" width="14" height="10" rx="2" fill="var({color})"{fade}/>')
        parts.append(f'{mark}<text x="{lx + 19}" y="16" font-size="11" fill="var(--text-secondary)">{label}</text>')
        lx += 26 + 6.2 * len(label)
    for level, label in [(support, "Support"), (resistance, "Resistance")]:
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(level):.1f}" y2="{y(level):.1f}" '
                     f'stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="4 3"/>'
                     f'<text x="{width - pad_r + 6}" y="{y(level) + 4:.1f}" font-size="11" fill="var(--text-muted)">'
                     f'{label} {level:,.1f}</text>')
    for k in range(5):
        value = low + (high - low) * k / 4
        parts.append(f'<text x="{pad_l - 6}" y="{y(value) + 4:.1f}" text-anchor="end" font-size="10" '
                     f'fill="var(--text-muted)">{value:,.0f}</text>')
    body = max(step * 0.6, 1.5)
    for k, row in enumerate(tail.itertuples()):
        color = "var(--good)" if row.close >= row.open else "var(--critical)"
        top, bottom = y(max(row.open, row.close)), y(min(row.open, row.close))
        parts.append(f'<g class="candle"><title>{row.Index:%Y-%m-%d}: open {row.open:,.1f}, high {row.high:,.1f}, '
                     f'low {row.low:,.1f}, close {row.close:,.1f}</title>'
                     f'<line x1="{x(k):.1f}" x2="{x(k):.1f}" y1="{y(row.high):.1f}" y2="{y(row.low):.1f}" '
                     f'stroke="{color}" stroke-width="1"/>'
                     f'<rect x="{x(k) - body / 2:.1f}" y="{top:.1f}" width="{body:.1f}" height="{max(bottom - top, 1):.1f}" '
                     f'fill="{color}" rx="1"/></g>')
    for series, label, color in [(ma20, "20-day avg", "--series-1"), (ma50, "50-day avg", "--series-2")]:
        points = [(x(k), y(v)) for k, v in enumerate(series) if pd.notna(v)]
        if points:
            path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(points))
            parts.append(f'<path d="{path}" fill="none" stroke="var({color})" stroke-width="2"/>')
    for k in sorted({0, len(tail) // 2, len(tail) - 1}):
        parts.append(f'<text x="{x(k):.1f}" y="{height - 6}" text-anchor="middle" font-size="10" '
                     f'fill="var(--text-muted)">{tail.index[k]:%d %b}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Candlestick chart, last {days} trading days">'
            f'{"".join(parts)}</svg>')


def _pick_cards(picks, details):
    cards = []
    for rank, p in enumerate(picks.itertuples(), 1):
        info = (details or {}).get(p.symbol, {})
        bullets = lambda items: "<ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in items) + "</ul>"
        cards.append(
            f'<div class="card pick"><h3>{rank}. {html.escape(p.symbol)} · {html.escape(p.sector)}</h3>'
            f'<p class="facts">Last close Rs {p.price:,.1f} · <b>buy between Rs {p.buy_low:,.1f} – {p.buy_high:,.1f}</b> · '
            f'profit growth {_pct(p.eps_growth, 0)} · P/E {p.pe:.1f} · price/book {p.pb:.2f}</p>'
            f'<div class="candles">{info.get("chart", "")}</div>'
            f'<div class="label">Why</div>{bullets(info.get("why", []))}'
            f'<div class="label">Risks</div>{bullets(info.get("risks", []))}</div>')
    return "".join(cards)


def _table(headers, rows):
    head = "".join(f'<th class="num">{h[1:]}</th>' if h.startswith("#") else f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(rows_cell for rows_cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _cell(value, numeric=False):
    return f'<td class="num">{value}</td>' if numeric else f"<td>{html.escape(str(value))}</td>"


def _mine_section(mine):
    """Your real holdings (local page only)."""
    if not mine:
        return ""
    t = mine["total"]
    tiles = [("Paid", f"Rs {t['cost']:,.0f}"), ("Worth now", f"Rs {t['value']:,.0f}"),
             ("Gain / loss", f"{t['gain']:+,.0f} ({_pct(t['gain_pct'])})"), ("Today", _move(t["today_pct"])),
             ("Take-home if sold today", f"Rs {t['take_home']:,.0f}")]
    rows = []
    for r in mine["rows"].itertuples():
        if pd.isna(r.value):
            rows.append([_cell(r.symbol), _cell(r.sector), _cell(f"{r.units}", True), _cell(f"{r.buy_price:,.1f}", True),
                         _cell("no price data", True), _cell("", True), _cell("", True), _cell("", True), _cell("")])
            continue
        notes = " ".join(mine["notes"].get(r.symbol, [])) + (f" [{r.note}]" if r.note else "")
        rows.append([_cell(r.symbol), _cell(r.sector), _cell(f"{r.units}", True), _cell(f"{r.buy_price:,.1f}", True),
                     _cell(f"{r.last:,.1f}", True), _cell(f"{r.value:,.0f}", True),
                     _cell(f"{r.gain:+,.0f} ({_pct(r.gain_pct)})", True), _cell(_move(r.day_change), True), _cell(notes)])
    unpriced = f"<p class='note'>No price data for: {', '.join(t['unpriced'])}.</p>" if t["unpriced"] else ""
    return (f'<h2>My real portfolio</h2><div class="tiles">'
            + "".join(f'<div class="tile"><div class="label">{l}</div><div class="value">{v}</div></div>' for l, v in tiles)
            + '</div><div class="card" style="margin-top:12px">'
            + _table(["Share", "Sector", "#Units", "#Bought at", "#Last", "#Value (Rs)", "#Gain / loss", "#Today", "Health check"], rows)
            + f'{unpriced}<p class="note">Take-home: after broker fees, SEBON fee, DP charge and 10% tax on any profit. '
              'Only on this computer — never published.</p></div>')


def render(state, equity, holdings, picks, movers, next_day, details=None, record=None, mine=None):
    """The whole page as an HTML string."""
    value = equity["portfolio"].iloc[-1]
    paper_return = value / equity["portfolio"].iloc[0] - 1
    market = equity["index"].iloc[-1] / equity["index"].iloc[0] - 1
    slots = (f"{round(state['top_k'] * state['exposure'])} of {state['top_k']}"
             if state.get("exposure") is not None else f"{state['top_k']} of {state['top_k']}")
    tiles = [("Portfolio value", f"Rs {value:,.0f}"), ("Since start", _move(paper_return)),
             ("NEPSE index, same days", _move(market)), ("Ahead of the market", f"{(paper_return - market) * 100:+.1f} pts"),
             ("Money conditions", f"invest {slots} slots")]

    holding_rows = []
    for h in holdings.sort_values("symbol").itertuples():
        holding_rows.append([_cell(h.symbol), _cell(h.sector), _cell(f"{h.units}", True), _cell(f"{h.bought:,.1f}", True),
                             _cell(f"{h.last:,.1f}", True), _cell(f"{h.units * h.last:,.0f}", True),
                             _cell(_move(h.last / h.bought - 1), True)])
    orders = [f"<li>SELL all of {html.escape(s)} — {html.escape(r)}</li>" for s, r in sorted(state["sell_orders"].items())]
    orders += [f"<li>BUY {html.escape(s)} for about Rs {b:,.0f}</li>" for s, b in state["buy_orders"]]

    pick_rows = [[_cell(p.symbol), _cell(p.sector), _cell(f"{p.price:,.1f}", True),
                  _cell(f"Rs {p.buy_low:,.1f} – {p.buy_high:,.1f}", True), _cell(_pct(p.eps_growth, 0), True),
                  _cell(f"{p.pe:.1f}", True), _cell(f"{p.pb:.2f}", True)] for p in picks.itertuples()]
    mover_table = lambda frame: _table(["Share", "Sector", "#Close", "#Change"], [
        [_cell(m.symbol), _cell(m.sector), _cell(f"{m.close:,.1f}", True), _cell(_move(m.change), True)]
        for m in frame.itertuples()])
    gainers = movers[movers["change"] > 0].sort_values("change", ascending=False).head(5)
    losers = movers[movers["change"] < 0].sort_values("change").head(5)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>NEPSE dashboard {state['date']}</title>
<style>{STYLE}</style></head><body><main class="viz-root">
<h1>NEPSE dashboard — {state['date']}</h1>
<p class="sub">Paper portfolio started {state['start']} with Rs {state['capital']:,.0f} (pretend money).
Rules: calmest shares + {html.escape(', '.join(state.get('features') or []) or 'nothing extra')}.</p>
{_mine_section(mine)}
<h2>Paper portfolio</h2>
<div class="tiles">{''.join(f'<div class="tile"><div class="label">{l}</div><div class="value">{v}</div></div>' for l, v in tiles)}</div>

<h2>Portfolio vs market (both start at 100)</h2>
<div class="card"><div class="legend"><span><span class="swatch" style="background:var(--series-1)"></span>Your portfolio</span>
<span><span class="swatch" style="background:var(--series-2)"></span>NEPSE index</span></div>
<div class="chart">{line_chart(equity)}<div class="tip"></div></div></div>

<h2>Holdings ({len(holdings)} shares · cash Rs {state['cash']:,.0f})</h2>
<div class="card">{_table(["Share", "Sector", "#Units", "#Bought at", "#Last", "#Value (Rs)", "#Change"], holding_rows)}</div>

<h2>Orders for the next trading morning ({next_day:%a %Y-%m-%d})</h2>
<div class="card">{'<ul>' + ''.join(orders) + '</ul>' if orders else '<p>Nothing to do.</p>'}
<p class="note">Next ranking refresh in {state['next_refresh_in']} trading day(s).</p></div>

<h2>Top 5 to consider for {next_day:%a %Y-%m-%d} (at these prices)</h2>
<div class="card">{_table(["Share", "Sector", "#Last close", "#Buy between", "#Profit growth", "#P/E", "#Price/book"], pick_rows)
                   if pick_rows else '<p>No share passes the screen today.</p>'}
<p class="note">Screen: business (profit per share up &gt;10% on a year ago, P/E ≤ 25, price ≤ 2.5× book value, passes the
quality and new-supply filters) + chart (not in a falling trend, not overheated) + big brokers not clearly selling; calmest first.
Use a limit order inside the range and don't chase above it. {_record_text(record)} This is not investment advice.</p></div>
{_pick_cards(picks, details)}

<h2>Today's top movers (150 most-traded shares)</h2>
<div class="cols"><div class="card"><b>Gainers</b>{mover_table(gainers)}</div>
<div class="card"><b>Losers</b>{mover_table(losers)}</div></div>
<p class="note">Built from community datasets and the NEPSE website after the close. Research and analysis only —
not investment advice.</p>
</main><script>{SCRIPT}</script></body></html>"""


def _record_text(record):
    if not record:
        return ""
    return (f"Track record of this screen ({record['picks']} past picks, each held one year, late 2022–2025): "
            f"beat the market {record['beat']:.0%} of the time, average {_pct(record['avg'])} vs the market's "
            f"{_pct(record['market'])}, made money {record['positive']:.0%} of the time — sound, fairly priced shares, "
            "not predictions.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-dir", default="outputs/nepse/paper")
    parser.add_argument("--clean-dir", default="data/nepse/clean_full")
    parser.add_argument("--meta-dir", default="data/nepse/meta")
    parser.add_argument("--out", default="outputs/nepse/dashboard.html")
    parser.add_argument("--open", action="store_true", help="open the page in the browser")
    parser.add_argument("--record", default="outputs/nepse/picks_track_record.csv",
                        help="cached replay of the screen on past data (rebuilt with --refresh-record)")
    parser.add_argument("--refresh-record", action="store_true")
    parser.add_argument("--holdings", default=str(HOLDINGS), help="your real holdings (shown only on the local page)")
    parser.add_argument("--public-out", default="outputs/nepse/dashboard_public.html",
                        help="copy without your real holdings, for publishing")
    args = parser.parse_args(argv)

    paper, meta = Path(args.paper_dir), Path(args.meta_dir)
    state = json.loads((paper / "state.json").read_text())
    equity = pd.read_csv(paper / "equity.csv", parse_dates=["timestamps"], index_col="timestamps")
    trades = pd.read_csv(paper / "trades.csv")
    sectors = load_sectors(meta)
    stocks = tradable_stocks(load_panel(args.clean_dir, include_indices=True), sectors)
    latest = pd.Timestamp(state["date"])

    bought = trades[trades["side"] == "buy"].groupby("symbol")["price"].last()
    holdings = pd.DataFrame([{"symbol": s, "sector": sectors.get(s, ""), "units": n, "bought": bought.get(s, float("nan")),
                              "last": stocks[s]["close"].loc[:latest].iloc[-1]}
                             for s, n in state["holdings"].items()],
                            columns=["symbol", "sector", "units", "bought", "last"])

    reports = pd.read_csv(meta / "fundamentals.csv")
    blocks = [quality_blocklist(reports, stocks),
              supply_blocklist(pd.read_csv(meta / "lockins.csv"), pd.read_csv(meta / "rights.csv"))]
    broker_file = meta / "broker_features.csv.gz"
    broker = pd.read_csv(broker_file, parse_dates=["date"]) if broker_file.exists() else None
    blocklist = lambda day: set().union(*(b(day) for b in blocks))
    picks = daily_picks(stocks, sectors, reports, latest, broker=broker, blocklist=blocklist)

    actions, rights = pd.read_csv(meta / "corporate_actions.csv"), pd.read_csv(meta / "rights.csv")
    lockins = pd.read_csv(meta / "lockins.csv")
    details = {}
    for pick in picks.itertuples():
        why, risks = explain(pick, actions, rights, latest, lockins)
        chart = candle_chart(stocks[pick.symbol].loc[:latest], pick.buy_low, pick.buy_high, pick.support, pick.resistance)
        details[pick.symbol] = {"why": why, "risks": risks, "chart": chart}

    record_file = Path(args.record)
    if args.refresh_record or not record_file.exists():
        panel_index = load_panel(args.clean_dir, include_indices=True)["NEPSE_INDEX"]["close"]
        yields = pd.read_csv(meta / "market_dividend_yield.csv", index_col="year")["dividend_yield"]
        past = track_record(stocks, reports, panel_index.index, total_return_index(panel_index, yields),
                            "2022-10-01", f"{latest - pd.Timedelta(days=370):%Y-%m-%d}", broker=broker, blocklist=blocklist)
        record_file.parent.mkdir(parents=True, exist_ok=True)
        past.to_csv(record_file, index=False)
    past = pd.read_csv(record_file)
    record = ({"picks": len(past), "beat": (past["return"] > past["market"]).mean(), "avg": past["return"].mean(),
               "market": past["market"].mean(), "positive": (past["return"] > 0).mean()} if len(past) else None)

    rows = []
    for symbol in liquid_universe(stocks, latest, top_n=150, min_history=20):
        close = stocks[symbol]["close"].loc[:latest]
        rows.append({"symbol": symbol, "sector": sectors.get(symbol, ""), "close": close.iloc[-1],
                     "change": close.iloc[-1] / close.iloc[-2] - 1})
    movers = pd.DataFrame(rows, columns=["symbol", "sector", "close", "change"])

    next_day = next_trading_days(latest, 1, load_holidays()).iloc[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mine = None
    if Path(args.holdings).exists():
        valued = value_holdings(load_holdings(args.holdings), stocks, latest, sectors)
        mine = {"rows": valued, "total": summarize(valued, CostModel()),
                "notes": {s: health_notes(s, stocks, reports, latest) for s in valued["symbol"]}}
    out.write_text(render(state, equity, holdings, picks, movers, next_day, details, record, mine))
    Path(args.public_out).write_text(render(state, equity, holdings, picks, movers, next_day, details, record))
    print(f"Wrote {out} (with your holdings) and {args.public_out} (without)")
    if args.open:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
