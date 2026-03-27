"""FastAPI dashboard for KalshiEdge metrics."""

import datetime
import time

import structlog
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from kalshiedge.calibration import brier_score
from kalshiedge.config import settings
from kalshiedge.kalshi_client import KalshiClient
from kalshiedge.portfolio import PortfolioStore

logger = structlog.get_logger()

app = FastAPI(title="KalshiEdge Dashboard")
store = PortfolioStore()
kalshi = KalshiClient()
_start_time = time.time()


@app.on_event("startup")
async def startup():
    await store.initialize()


@app.on_event("shutdown")
async def shutdown():
    await store.close()
    await kalshi.close()


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    uptime = time.time() - _start_time
    total_trades = await store.execute_fetchall("SELECT COUNT(*) FROM trades")
    total_forecasts = await store.execute_fetchall("SELECT COUNT(*) FROM forecasts")
    positions = await store.get_open_positions()
    bankroll = await store.get_bankroll_cents()
    return {
        "status": "healthy",
        "uptime_seconds": int(uptime),
        "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "env": settings.kalshi_env,
        "dry_run": settings.dry_run,
        "bankroll_usd": bankroll / 100 if bankroll else 0,
        "open_positions": len(positions),
        "total_trades": total_trades[0][0],
        "total_forecasts": total_forecasts[0][0],
    }


@app.get("/api/live")
async def get_live():
    """Fetch real-time data from Kalshi production API."""
    result = {
        "env": settings.kalshi_env,
        "dry_run": settings.dry_run,
        "balance_cents": None,
        "balance_usd": None,
        "positions_value_usd": None,
        "portfolio_total_usd": None,
        "kalshi_positions": [],
        "error": None,
    }
    try:
        bal = await kalshi.get_balance()
        cents = bal.get("balance", 0)
        result["balance_cents"] = cents
        result["balance_usd"] = cents / 100
    except Exception as e:
        result["error"] = str(e)
        logger.warning("live_balance_fetch_failed", error=str(e))

    try:
        pos = await kalshi.get_positions()
        positions = pos.get("market_positions", pos.get("positions", []))
        pos_list = []
        total_value_cents = 0
        for p in positions:
            qty = _pos_qty(p)
            if qty <= 0:
                continue
            # market_exposure is in dollars as string or cents as int
            exposure = p.get("market_exposure", 0)
            if isinstance(exposure, str):
                exposure_cents = int(round(float(exposure) * 100))
            else:
                exposure_cents = int(exposure)
            total_value_cents += exposure_cents
            pos_list.append({
                "ticker": p.get("ticker", ""),
                "side": _pos_side(p),
                "quantity": qty,
                "market_value_usd": exposure_cents / 100,
            })
        result["kalshi_positions"] = pos_list
        result["positions_value_usd"] = total_value_cents / 100
        cash = result["balance_usd"] or 0
        result["portfolio_total_usd"] = cash + total_value_cents / 100
    except Exception as e:
        if not result["error"]:
            result["error"] = str(e)
        logger.warning("live_positions_fetch_failed", error=str(e))

    return result


def _pos_side(p: dict) -> str:
    yes = p.get("yes_contracts", p.get("total_traded", 0))
    no = p.get("no_contracts", 0)
    if isinstance(yes, (int, float)) and yes > 0:
        return "yes"
    if isinstance(no, (int, float)) and no > 0:
        return "no"
    return "yes"


def _pos_qty(p: dict) -> int:
    yes = p.get("yes_contracts", 0)
    no = p.get("no_contracts", 0)
    try:
        return max(int(yes or 0), int(no or 0))
    except (ValueError, TypeError):
        return 0


@app.get("/api/metrics")
async def get_metrics():
    bankroll = await store.get_bankroll_cents()
    daily_pnl = await store.get_daily_pnl_cents()
    positions = await store.get_open_positions()

    trades = await store.execute_fetchall(
        "SELECT ticker, side, action, price_cents, count, status, created_at, pnl_cents "
        "FROM trades ORDER BY created_at DESC LIMIT 20"
    )

    forecasts = await store.execute_fetchall(
        "SELECT ticker, title, market_price_cents, model_probability, edge, reasoning, "
        "created_at, actual_outcome "
        "FROM forecasts ORDER BY created_at DESC LIMIT 20"
    )

    resolved = await store.execute_fetchall(
        "SELECT model_probability, actual_outcome FROM forecasts "
        "WHERE actual_outcome IS NOT NULL"
    )
    resolved_pairs = [(r[0], r[1]) for r in resolved]
    brier = brier_score(resolved_pairs) if resolved_pairs else None

    total_trades = await store.execute_fetchall("SELECT COUNT(*) FROM trades")
    total_forecasts = await store.execute_fetchall("SELECT COUNT(*) FROM forecasts")
    total_resolved = len(resolved_pairs)

    today = datetime.date.today().isoformat()
    trades_today = await store.execute_fetchall(
        "SELECT COUNT(*) FROM trades WHERE date(created_at) = ?", (today,)
    )

    # Strategy performance
    strategy_stats = await store.execute_fetchall(
        """SELECT
            COALESCE(strategy, 'calibration_edge') as strat,
            COUNT(*) as total,
            SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_cents < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN pnl_cents IS NULL THEN 1 ELSE 0 END) as pending,
            COALESCE(SUM(pnl_cents), 0) as total_pnl
        FROM trades GROUP BY strat"""
    )

    return {
        "bankroll_cents": bankroll,
        "bankroll_usd": bankroll / 100 if bankroll else 0,
        "daily_pnl_cents": daily_pnl,
        "daily_pnl_usd": daily_pnl / 100,
        "open_positions": len(positions),
        "positions": [
            {
                "ticker": p["ticker"],
                "side": p["side"],
                "price_cents": p["price_cents"],
                "count": p["count"],
                "status": p["status"],
            }
            for p in positions
        ],
        "recent_trades": [
            {
                "ticker": t[0], "side": t[1], "action": t[2], "price_cents": t[3],
                "count": t[4], "status": t[5], "created_at": t[6], "pnl_cents": t[7],
            }
            for t in trades
        ],
        "recent_forecasts": [
            {
                "ticker": f[0], "title": f[1], "market_price_cents": f[2],
                "model_probability": f[3], "edge": f[4], "reasoning": f[5],
                "created_at": f[6], "resolved": f[7] is not None, "outcome": f[7],
            }
            for f in forecasts
        ],
        "brier_score": round(brier, 4) if brier is not None else None,
        "total_trades": total_trades[0][0],
        "total_forecasts": total_forecasts[0][0],
        "total_resolved": total_resolved,
        "trades_today": trades_today[0][0],
        "strategies": [
            {
                "name": s[0],
                "total_trades": s[1],
                "wins": s[2] or 0,
                "losses": s[3] or 0,
                "pending": s[4] or 0,
                "pnl_cents": s[5],
                "pnl_usd": s[5] / 100,
            }
            for s in strategy_stats
        ],
    }


@app.get("/api/charts")
async def get_charts():
    """Chart data for balance history and daily P&L."""
    snapshots = await store.execute_fetchall(
        "SELECT date, balance_cents, pnl_cents FROM daily_snapshots ORDER BY date"
    )
    # Rolling Brier (last 20 resolved)
    recent_resolved = await store.execute_fetchall(
        "SELECT model_probability, actual_outcome FROM forecasts "
        "WHERE actual_outcome IS NOT NULL ORDER BY created_at DESC LIMIT 20"
    )
    rolling_pairs = [(r[0], r[1]) for r in recent_resolved]
    rolling_brier = brier_score(rolling_pairs) if rolling_pairs else None

    # Max drawdown
    balances = [s[1] for s in snapshots] if snapshots else []
    max_dd = 0.0
    peak = 0
    for b in balances:
        if b > peak:
            peak = b
        if peak > 0:
            dd = (peak - b) / peak
            if dd > max_dd:
                max_dd = dd

    return {
        "balance_history": [
            {"date": s[0], "balance_usd": s[1] / 100}
            for s in snapshots
        ],
        "pnl_history": [
            {"date": s[0], "pnl_usd": s[2] / 100}
            for s in snapshots
        ],
        "rolling_brier": round(rolling_brier, 4) if rolling_brier is not None else None,
        "max_drawdown_pct": round(max_dd * 100, 1),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KalshiEdge Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --muted: #8b8fa3; --accent: #6c5ce7;
    --green: #00b894; --red: #e17055; --yellow: #fdcb6e;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg); color: var(--text); padding: 24px;
    min-height: 100vh;
  }
  h1 { font-size: 1.4rem; margin-bottom: 8px; }
  .subtitle { color: var(--muted); font-size: 0.8rem; margin-bottom: 24px; }
  .mode-live { color: var(--red); font-weight: 600; }
  .mode-dry { color: var(--yellow); font-weight: 600; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .card-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .card-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .neutral { color: var(--yellow); }
  h2 { font-size: 1rem; margin-bottom: 12px; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px 12px;
       border-bottom: 1px solid var(--border); font-size: 0.7rem; text-transform: uppercase; }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
  tr:hover { background: rgba(108, 92, 231, 0.05); }
  .section { margin-bottom: 32px; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 600;
  }
  .badge-yes { background: rgba(0,184,148,0.15); color: var(--green); }
  .badge-no { background: rgba(225,112,85,0.15); color: var(--red); }
  .badge-pending { background: rgba(253,203,110,0.15); color: var(--yellow); }
  .badge-executed { background: rgba(0,184,148,0.15); color: var(--green); }
  .badge-resting { background: rgba(253,203,110,0.15); color: var(--yellow); }
  .edge-bar {
    display: inline-block; height: 6px; border-radius: 3px;
    background: var(--accent); min-width: 4px;
  }
  .refresh-note { color: var(--muted); font-size: 0.7rem; text-align: right; }
  .empty { color: var(--muted); font-style: italic; padding: 24px; text-align: center; }
  .error-banner { background: rgba(225,112,85,0.1); border: 1px solid var(--red); border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; color: var(--red); font-size: 0.8rem; display: none; }
</style>
</head>
<body>

<h1>KalshiEdge</h1>
<p class="subtitle">Autonomous Kalshi Trading Agent &mdash; <span id="env">--</span> &mdash; <span id="updated"></span></p>
<div class="error-banner" id="error-banner"></div>

<div class="grid" id="cards">
  <div class="card"><div class="card-label">Portfolio Total</div><div class="card-value" id="portfolio-total">--</div></div>
  <div class="card"><div class="card-label">Cash</div><div class="card-value" id="balance">--</div></div>
  <div class="card"><div class="card-label">In Positions</div><div class="card-value" id="positions-value">--</div></div>
  <div class="card"><div class="card-label">Daily P&amp;L</div><div class="card-value" id="pnl">--</div></div>
  <div class="card"><div class="card-label">Trades Today</div><div class="card-value" id="trades-today">--</div></div>
  <div class="card"><div class="card-label">Brier Score</div><div class="card-value" id="brier">--</div></div>
  <div class="card"><div class="card-label">Total Forecasts</div><div class="card-value" id="total-forecasts">--</div></div>
  <div class="card"><div class="card-label">Rolling Brier (20)</div><div class="card-value" id="rolling-brier">--</div></div>
  <div class="card"><div class="card-label">Max Drawdown</div><div class="card-value" id="max-drawdown">--</div></div>
</div>

<div class="section" style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div class="card"><h2 style="margin-bottom:8px">Balance History</h2><canvas id="balance-chart" height="200"></canvas></div>
  <div class="card"><h2 style="margin-bottom:8px">Daily P&amp;L</h2><canvas id="pnl-chart" height="200"></canvas></div>
</div>

<div class="section">
  <h2>Kalshi Positions (Live)</h2>
  <div class="card">
    <table><thead><tr><th>Ticker</th><th>Side</th><th>Qty</th><th>Value</th></tr></thead>
    <tbody id="kalshi-positions-table"></tbody></table>
  </div>
</div>

<div class="section">
  <h2>Recent Forecasts</h2>
  <div class="card">
    <table><thead><tr><th>Time</th><th>Ticker</th><th>Market</th><th>Model</th><th>Edge</th><th>Outcome</th></tr></thead>
    <tbody id="forecasts-table"></tbody></table>
  </div>
</div>

<div class="section">
  <h2>Recent Trades</h2>
  <div class="card">
    <table><thead><tr><th>Time</th><th>Ticker</th><th>Side</th><th>Price</th><th>Qty</th><th>Status</th><th>P&amp;L</th></tr></thead>
    <tbody id="trades-table"></tbody></table>
  </div>
</div>

<div class="section">
  <h2>Strategy Performance</h2>
  <div class="card">
    <table><thead><tr><th>Strategy</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Pending</th><th>P&amp;L</th></tr></thead>
    <tbody id="strategy-table"></tbody></table>
  </div>
</div>

<p class="refresh-note">Auto-refreshes every 30s</p>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
function fmt$(cents) {
  if (cents == null) return '--';
  const d = cents / 100;
  return (d >= 0 ? '+' : '') + '$' + Math.abs(d).toFixed(2);
}
function badge(text, cls) { return `<span class="badge badge-${cls}">${text}</span>`; }
function pct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '--'; }
function shortTime(ts) {
  if (!ts) return '--';
  const d = new Date(ts);
  return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

async function refresh() {
  try {
    const [mRes, lRes] = await Promise.all([
      fetch('/api/metrics'),
      fetch('/api/live'),
    ]);
    const d = await mRes.json();
    const live = await lRes.json();

    // Env / mode
    const envEl = document.getElementById('env');
    if (live.dry_run) {
      envEl.innerHTML = `<span class="mode-dry">${live.env.toUpperCase()} (DRY RUN)</span>`;
    } else {
      envEl.innerHTML = `<span class="mode-live">${live.env.toUpperCase()} (LIVE)</span>`;
    }

    // Error banner
    const errEl = document.getElementById('error-banner');
    if (live.error) {
      errEl.textContent = 'Kalshi API: ' + live.error;
      errEl.style.display = 'block';
    } else {
      errEl.style.display = 'none';
    }

    // Balance — prefer live Kalshi balance
    const balUsd = live.balance_usd != null ? live.balance_usd : d.bankroll_usd;
    document.getElementById('balance').textContent = '$' + balUsd.toFixed(2);

    const posVal = live.positions_value_usd != null ? live.positions_value_usd : 0;
    document.getElementById('positions-value').textContent = '$' + posVal.toFixed(2);

    const totalVal = live.portfolio_total_usd != null ? live.portfolio_total_usd : balUsd;
    document.getElementById('portfolio-total').textContent = '$' + totalVal.toFixed(2);

    const pnlEl = document.getElementById('pnl');
    pnlEl.textContent = fmt$(d.daily_pnl_cents);
    pnlEl.className = 'card-value ' + (d.daily_pnl_cents > 0 ? 'positive' : d.daily_pnl_cents < 0 ? 'negative' : '');

    document.getElementById('trades-today').textContent = d.trades_today;
    document.getElementById('brier').textContent = d.brier_score != null ? d.brier_score.toFixed(4) : 'N/A';
    document.getElementById('total-forecasts').textContent = d.total_forecasts + (d.total_resolved > 0 ? ` (${d.total_resolved} resolved)` : '');
    document.getElementById('updated').textContent = new Date().toLocaleTimeString();

    // Kalshi live positions
    const kpt = document.getElementById('kalshi-positions-table');
    if (live.kalshi_positions.length === 0) {
      kpt.innerHTML = '<tr><td colspan="4" class="empty">No open positions on Kalshi</td></tr>';
    } else {
      kpt.innerHTML = live.kalshi_positions.map(p => `<tr>
        <td>${p.ticker}</td>
        <td>${badge(p.side.toUpperCase(), p.side)}</td>
        <td>${p.quantity}</td>
        <td>$${(p.market_value_usd || 0).toFixed(2)}</td>
      </tr>`).join('');
    }

    // Forecasts
    const ft = document.getElementById('forecasts-table');
    if (d.recent_forecasts.length === 0) {
      ft.innerHTML = '<tr><td colspan="6" class="empty">No forecasts yet &mdash; run the agent to generate forecasts</td></tr>';
    } else {
      ft.innerHTML = d.recent_forecasts.map(f => {
        const edgeW = Math.min(Math.abs(f.edge) * 500, 100);
        return `<tr>
          <td>${shortTime(f.created_at)}</td>
          <td>${f.ticker}</td>
          <td>${f.market_price_cents}&cent;</td>
          <td>${pct(f.model_probability)}</td>
          <td><span class="edge-bar" style="width:${edgeW}px"></span> ${pct(f.edge)}</td>
          <td>${f.resolved ? (f.outcome === 1 ? badge('YES','yes') : badge('NO','no')) : '--'}</td>
        </tr>`;
      }).join('');
    }

    // Trades
    const tt = document.getElementById('trades-table');
    if (d.recent_trades.length === 0) {
      tt.innerHTML = '<tr><td colspan="7" class="empty">No trades yet</td></tr>';
    } else {
      tt.innerHTML = d.recent_trades.map(t => {
        const pnlClass = t.pnl_cents > 0 ? 'positive' : t.pnl_cents < 0 ? 'negative' : '';
        return `<tr>
          <td>${shortTime(t.created_at)}</td>
          <td>${t.ticker}</td>
          <td>${badge(t.side.toUpperCase(), t.side)}</td>
          <td>${t.price_cents}&cent;</td>
          <td>${t.count}</td>
          <td>${badge(t.status, t.status === 'executed' ? 'executed' : t.status)}</td>
          <td class="${pnlClass}">${t.pnl_cents != null ? fmt$(t.pnl_cents) : '--'}</td>
        </tr>`;
      }).join('');
    }
    // Strategy performance
    const st = document.getElementById('strategy-table');
    if (!d.strategies || d.strategies.length === 0) {
      st.innerHTML = '<tr><td colspan="6" class="empty">No strategy data yet</td></tr>';
    } else {
      st.innerHTML = d.strategies.map(s => {
        const pnlClass = s.pnl_cents > 0 ? 'positive' : s.pnl_cents < 0 ? 'negative' : '';
        const words = s.name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1));
        const name = words.join(' ');
        return `<tr>
          <td>${name}</td>
          <td>${s.total_trades}</td>
          <td class="positive">${s.wins}</td>
          <td class="negative">${s.losses}</td>
          <td class="neutral">${s.pending}</td>
          <td class="${pnlClass}">${fmt$(s.pnl_cents)}</td>
        </tr>`;
      }).join('');
    }

  } catch(e) { console.error('Refresh failed:', e); }
}

let balChart = null, pnlChart = null;
const chartOpts = {responsive:true, plugins:{legend:{display:false}}, scales:{x:{ticks:{color:'#8b8fa3',font:{size:10}}}, y:{ticks:{color:'#8b8fa3'}}}};

async function refreshCharts() {
  try {
    const r = await fetch('/api/charts');
    const c = await r.json();

    // Rolling Brier + drawdown
    document.getElementById('rolling-brier').textContent = c.rolling_brier != null ? c.rolling_brier.toFixed(4) : 'N/A';
    const ddEl = document.getElementById('max-drawdown');
    ddEl.textContent = c.max_drawdown_pct + '%';
    ddEl.className = 'card-value ' + (c.max_drawdown_pct > 20 ? 'negative' : c.max_drawdown_pct > 10 ? 'neutral' : 'positive');

    // Balance chart
    if (c.balance_history.length > 0) {
      const labels = c.balance_history.map(d => d.date.slice(5));
      const data = c.balance_history.map(d => d.balance_usd);
      if (balChart) balChart.destroy();
      balChart = new Chart(document.getElementById('balance-chart'), {
        type: 'line', data: {labels, datasets: [{data, borderColor:'#6c5ce7', backgroundColor:'rgba(108,92,231,0.1)', fill:true, tension:0.3, pointRadius:2}]}, options: chartOpts
      });
    }

    // P&L chart
    if (c.pnl_history.length > 0) {
      const labels = c.pnl_history.map(d => d.date.slice(5));
      const data = c.pnl_history.map(d => d.pnl_usd);
      const colors = data.map(v => v >= 0 ? '#00b894' : '#e17055');
      if (pnlChart) pnlChart.destroy();
      pnlChart = new Chart(document.getElementById('pnl-chart'), {
        type: 'bar', data: {labels, datasets: [{data, backgroundColor:colors}]}, options: chartOpts
      });
    }
  } catch(e) { console.error('Chart refresh failed:', e); }
}

refresh();
refreshCharts();
setInterval(refresh, 30000);
setInterval(refreshCharts, 60000);
</script>
</body>
</html>
"""
