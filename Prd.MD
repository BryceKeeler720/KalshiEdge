# PRD: Autonomous Kalshi Trading Agent ("KalshiEdge")

## Overview

KalshiEdge is a fully autonomous Python AI agent that discovers, evaluates, and trades prediction market contracts on Kalshi — the only CFTC-regulated prediction market exchange in the United States. It uses Claude (Anthropic API) as its forecasting engine, compares LLM-estimated probabilities against market prices to find mispricings, and executes trades via Kalshi's REST API — all wrapped in 2signal observability for full trace visibility. It runs 24/7 on a Proxmox LXC container with a $100 USD starting bankroll.

## Why Kalshi Over Polymarket

- **Fully legal for US residents** — CFTC Designated Contract Market, no VPN or geo-workarounds
- **USD settlement** — deposit/withdraw via ACH bank transfer (free), no crypto wallets or USDC needed
- **Single unified API** — one base URL for market data, trading, and portfolio (vs Polymarket's 3 APIs)
- **Free demo sandbox** — full-featured test environment at `demo-api.kalshi.co` with fake money
- **Simpler auth** — RSA-PSS signed requests, no blockchain wallet signatures
- **Zero fees on resting limit orders** — maker orders are fee-free
- **No heartbeat requirement** — resting orders don't auto-cancel

**Tradeoffs:** Kalshi has less total liquidity than Polymarket, fewer concurrent markets, and more restricted categories due to CFTC oversight. Prices are in cents (integers 1-99), not decimals.

## Goals

- Generate positive returns net of API costs and fees on a $100 USD bankroll
- Maintain full observability via 2signal traces on every agent decision
- Run autonomously on a Proxmox LXC with <1GB RAM, <$2/day API cost
- Paper trade for 50+ predictions in demo environment before going live

## Non-Goals

- Market making (requires significant capital)
- HFT/latency-sensitive strategies (FIX protocol, homelab latency too high)
- Cross-platform arbitrage (Kalshi ↔ Polymarket — different resolution rules)
- Building a UI dashboard (CLI + Telegram alerts only)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   KalshiEdge Agent                      │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Discovery│→ │ Research │→ │Forecaster│→ │ Trader │  │
│  │ (Markets)│  │ (News)   │  │ (Claude) │  │(Orders)│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│       ↑                                        │        │
│       └────── Risk Manager ◄───────────────────┘        │
│                    │                                    │
│              Portfolio State (SQLite)                   │
└─────────────────────────────────────────────────────────┘
         │                              │
    2signal traces                 Telegram alerts
```

### Components

**1. Market Discovery** — Polls `GET /markets` every 5 minutes. Filters by status, volume, liquidity, and time-to-expiry. Prioritizes new markets, volume spikes, and categories where LLMs have strongest edge (politics, economics, geopolitics).

**2. News Research** — For each candidate market, gathers news via Google News RSS (free), GDELT (free, 15-min updates), NewsAPI (500 free/day). Caches 30 min per market.

**3. Forecaster** — Claude Sonnet 4.6, 3 independent calls at temperatures 0.3/0.5/0.7, trimmed mean aggregation, extremization (factor 1.3) to correct LLM hedging bias. Wrapped in 2signal `wrap_anthropic`.

**4. Edge Detection** — Compares model probability to market price (cents/100). Minimum 8% edge threshold. Quarter-Kelly position sizing capped at 5% of bankroll.

**5. Order Execution** — `POST /portfolio/orders` with limit orders only (fee-free). Prices in cents (integer 1-99). Monitors fills via `GET /portfolio/orders`.

**6. Risk Manager** — Circuit breakers at 10% daily loss, 30% drawdown (halve sizes), 40% drawdown (hard stop). Max 12 concurrent positions, 50% total exposure, 5% per trade.

**7. Portfolio State** — SQLite with WAL mode. Tables for trades, positions, forecasts, daily P&L. Full audit trail for calibration analysis.

---

## Data Flow Per Cycle

```
1. Discovery:    GET /markets → candidate_markets[]
2. Filter:       candidate_markets → filtered_markets[] (risk + liquidity)
3. Research:     For each market: news APIs → context documents
4. Forecast:     For each market: Claude API (3x) → aggregated probability
5. Edge Check:   P_model vs P_market → trade_signals[] (edge > 8% only)
6. Risk Check:   trade_signals vs portfolio_state → approved_trades[]
7. Execute:      POST /portfolio/orders → order confirmations
8. Monitor:      GET /portfolio/orders → fill updates → state update
9. Log:          All steps traced via 2signal @observe decorators
```

---

## Trading Strategies (Priority Order)

**Strategy 1: Calibration Edge (Primary)** — Markets where Claude's estimate diverges >8% from price. Focus on new markets and retail-sentiment-heavy categories.

**Strategy 2: Near-Expiry Convergence** — Buy contracts priced 93-97¢ when resolution is imminent and outcome is certain. No settlement fees on Kalshi.

**Strategy 3: Event-Driven Positioning** — Position before Fed decisions, CPI releases, court rulings. Kalshi has strong economics markets ideal for this.

**Strategy 4: Intra-Event Arbitrage** — Multi-market events where outcome probabilities don't sum to ~100%.

---

## Fee Structure

- **Maker (limit) orders: FREE** — resting orders pay zero fees
- **Taker fees: ~1-2¢ per contract at 50¢, decreasing toward extremes**
- **No settlement fees** — $1.00 payout with no deductions
- **No API fees** — all endpoints free to call
- **Deposits: free via ACH** — 2% fee only on debit cards

---

## 2signal Integration

```python
from twosignal import TwoSignal, observe
from twosignal.types import SpanType
from twosignal.wrappers.anthropic import wrap_anthropic

ts = TwoSignal(api_key=os.getenv("TWOSIGNAL_API_KEY"))
anthropic_client = wrap_anthropic(anthropic.Anthropic())
```

Span mapping: `AGENT` → main loop, `RETRIEVAL` → discovery + news, `CHAIN` → forecast pipeline, `TOOL` → order execution, `GUARDRAIL` → risk checks. LLM spans auto-traced by `wrap_anthropic`.

---

## Configuration

```env
KALSHI_API_KEY_ID=               # RSA API key ID
KALSHI_PRIVATE_KEY_PATH=         # Path to RSA private key PEM
KALSHI_ENV=demo                  # "demo" or "prod"
ANTHROPIC_API_KEY=
TWOSIGNAL_API_KEY=
TWOSIGNAL_BASE_URL=
NEWSAPI_KEY=                     # Optional
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
BANKROLL_USD=100
MIN_EDGE_THRESHOLD=0.08
KELLY_FRACTION=0.25
MAX_POSITION_PCT=0.05
MAX_EXPOSURE_PCT=0.50
MAX_CONCURRENT_POSITIONS=12
CYCLE_INTERVAL_SECONDS=300
```

---

## Project Structure

```
kalshiedge/
├── CLAUDE.md
├── SKILLS.md
├── PRD.md
├── pyproject.toml
├── .env.example
├── src/
│   └── kalshiedge/
│       ├── __init__.py
│       ├── main.py              # Entry point, main loop
│       ├── config.py            # Pydantic settings from env
│       ├── kalshi_client.py     # Kalshi API client + RSA auth
│       ├── discovery.py         # Market discovery + filtering
│       ├── research.py          # News gathering pipeline
│       ├── forecaster.py        # Claude probability estimation
│       ├── edge.py              # Edge detection + Kelly sizing
│       ├── trader.py            # Order execution
│       ├── risk.py              # Risk manager + circuit breakers
│       ├── portfolio.py         # SQLite state management
│       ├── alerts.py            # Telegram notifications
│       ├── calibration.py       # Forecast vs outcome tracking
│       └── prompts.py           # Forecaster prompt templates
├── tests/
│   ├── test_discovery.py
│   ├── test_forecaster.py
│   ├── test_edge.py
│   ├── test_risk.py
│   └── test_portfolio.py
└── scripts/
    ├── calibration_report.py
    └── backtest.py
```

---

## Dependencies

```
kalshi-python>=2.1.4
anthropic>=0.50.0
twosignal>=0.1.0
httpx>=0.27.0
pydantic>=2.0
pydantic-settings>=2.0
python-dotenv>=1.0
pygooglenews>=0.1.0
gnews>=0.3.0
aiosqlite>=0.20.0
python-telegram-bot>=21.0
structlog>=24.0
cryptography>=43.0
```

---

## Milestones

**Phase 1: Foundation (Week 1-2)** — Project scaffold, Kalshi client with RSA auth, market discovery, news pipeline, Claude forecaster, SQLite state, edge detection.

**Phase 2: Demo Trading (Week 3-4)** — Full loop against `demo-api.kalshi.co`, 2signal tracing, Telegram alerts, calibration tracking, 50+ demo trades.

**Phase 3: Live Trading (Week 5-6)** — Switch to production, fund $100 via ACH, deploy to Proxmox LXC with systemd.

**Phase 4: Optimization (Ongoing)** — Tune extremization, Batch API for Claude costs, event calendar, WebSocket for real-time data.

---

## Success Criteria

- Demo: positive P&L over 50+ predictions
- Calibration: Brier score ≤ 0.15
- Live: net positive returns within 8 weeks
- Uptime: >95%, API costs <$2/day
- Observability: 100% of decisions traced in 2signal

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM forecasts worse than market | Lose bankroll | 50-trade demo period, calibration tracking, 40% drawdown stop |
| Kalshi API changes/downtime | Missed trades | Graceful degradation, backoff, Telegram alerts |
| Thin liquidity | Can't fill orders | Filter by orderbook depth, skip illiquid markets |
| State restrictions (sports in TX) | Limited access | Focus on politics, economics, geopolitics |
| API cost exceeds profits | Net loss | Batch API + prompt caching → <$1/day |