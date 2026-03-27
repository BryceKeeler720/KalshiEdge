# CLAUDE.md — KalshiEdge Agent

## What This Project Is

KalshiEdge is an autonomous Kalshi trading agent in Python. It discovers prediction markets via Kalshi's REST API, estimates true probabilities using Claude (Anthropic API) with news-augmented RAG, detects mispricings, and executes trades — all traced through 2signal observability. Runs on a Proxmox LXC container with a $100 USD bankroll.

Read `PRD.md` for full architecture and requirements. Read `SKILLS.md` for domain-specific technical reference.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run agent (demo mode by default)
python -m kalshiedge.main

# Run tests
pytest tests/ -v

# Calibration report
python scripts/calibration_report.py

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## Project Structure

```
src/kalshiedge/
├── main.py          # Entry point, async main loop
├── config.py        # Pydantic Settings from .env
├── kalshi_client.py # Kalshi REST API client + RSA-PSS auth
├── discovery.py     # Market discovery + filtering
├── research.py      # News pipeline (Google News, GDELT, NewsAPI)
├── forecaster.py    # Claude ensemble forecasting + extremization
├── edge.py          # Edge detection + quarter-Kelly sizing
├── trader.py        # Order execution via Kalshi API
├── risk.py          # Risk manager, circuit breakers, limits
├── portfolio.py     # SQLite state (trades, positions, forecasts)
├── alerts.py        # Telegram notifications
├── calibration.py   # Brier score tracking
└── prompts.py       # Superforecaster prompt templates
```

## Key Technical Decisions

### Kalshi API
- **Single unified API** — one base URL handles market data, trading, and portfolio
- **Base URLs**: Production `https://trading-api.kalshi.com/trade-api/v2`, Demo `https://demo-api.kalshi.co/trade-api/v2`
- **Auth**: RSA-PSS signed requests. Generate API key + download private key PEM from Kalshi dashboard. Each request is signed with `timestamp + method + path` using the RSA private key.
- **Prices are in CENTS as integers** — a price of 65 means $0.65 (65% probability). Range: 1-99.
- **Settlement**: USD, not crypto. Winners get $1.00 per contract ($100 in cents). Losers get $0.
- **Demo environment**: Full sandbox at `demo-api.kalshi.co` with fake money and real-ish market data. Always develop here first.

### Authentication Pattern
```python
from kalshi_python import Configuration, KalshiClient

config = Configuration(
    host="https://demo-api.kalshi.co/trade-api/v2"  # or production URL
)

with open(KALSHI_PRIVATE_KEY_PATH, "r") as f:
    private_key = f.read()

config.api_key_id = KALSHI_API_KEY_ID
config.private_key_pem = private_key

client = KalshiClient(config)
```

If using `kalshi-python` SDK has issues, fall back to raw `httpx` with manual RSA-PSS signing:
```python
import base64, time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def sign_request(private_key_pem: str, timestamp: str, method: str, path: str) -> str:
    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode()
    signature = key.sign(message, padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ), hashes.SHA256())
    return base64.b64encode(signature).decode()
```

### Order Execution Rules
- **Always use limit orders** — resting limit orders are fee-free on Kalshi. Taker (market) orders incur fees.
- Price in **integer cents** (1-99). `yes_price=55` means buy YES at 55¢.
- `action`: `"buy"` or `"sell"`
- `side`: `"yes"` or `"no"`
- `type`: `"limit"` (preferred) or `"market"`
- `count`: number of contracts (integer)
- `expiration_ts`: optional Unix timestamp for GTD orders (omit for GTC)
- **No heartbeat required** — unlike Polymarket, Kalshi orders persist until filled or cancelled
- Cancel with `DELETE /portfolio/orders/{order_id}`

### Forecaster Design
- Use `anthropic.Anthropic()` wrapped in `wrap_anthropic()` from 2signal
- Model: `claude-sonnet-4-6` for forecasting, `claude-haiku-4-5-20251001` for news screening
- Run 3 forecasts per market at temperatures 0.3, 0.5, 0.7
- Aggregate via trimmed mean, apply extremization (factor 1.3)
- **Convert Kalshi cents to probability before comparing**: `p_market = yes_price / 100`

### Risk Management Constants
```python
KELLY_FRACTION = 0.25           # Quarter Kelly
MAX_POSITION_PCT = 0.05         # 5% of bankroll per trade
MAX_EXPOSURE_PCT = 0.50         # 50% total exposure
MAX_CONCURRENT = 12             # Max open positions
MIN_EDGE = 0.08                 # 8% minimum edge to trade
DAILY_LOSS_LIMIT = 0.10         # 10% daily loss triggers pause
DRAWDOWN_REDUCE = 0.30          # At 30% drawdown, halve sizes
DRAWDOWN_STOP = 0.40            # At 40% drawdown, hard stop
```

### 2signal Integration
```python
from twosignal import TwoSignal, observe
from twosignal.types import SpanType
from twosignal.wrappers.anthropic import wrap_anthropic

ts = TwoSignal(
    api_key=os.getenv("TWOSIGNAL_API_KEY"),
    base_url=os.getenv("TWOSIGNAL_BASE_URL"),
    deployment_id="kalshiedge-v0.1.0",
)
anthropic_client = wrap_anthropic(anthropic.Anthropic())
```

Decorator mapping:
- `@observe(span_type=SpanType.AGENT)` → `run_cycle()`
- `@observe(span_type=SpanType.RETRIEVAL)` → `discover_markets()`, `gather_news()`
- `@observe(span_type=SpanType.CHAIN)` → `forecast_market()`
- `@observe(span_type=SpanType.TOOL)` → `place_order()`, `cancel_order()`, `get_balance()`
- `@observe(span_type=SpanType.GUARDRAIL)` → `check_risk_limits()`
- LLM spans auto-created by `wrap_anthropic`

Call `ts.shutdown()` on SIGTERM/SIGINT.

### SQLite Schema
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,         -- 'yes' or 'no'
    action TEXT NOT NULL,       -- 'buy' or 'sell'
    price_cents INTEGER NOT NULL,
    count INTEGER NOT NULL,
    order_id TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    filled_at TEXT,
    pnl_cents INTEGER
);

CREATE TABLE forecasts (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    market_price_cents INTEGER NOT NULL,
    model_probability REAL NOT NULL,
    confidence_low REAL,
    confidence_high REAL,
    edge REAL NOT NULL,
    reasoning TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    actual_outcome INTEGER     -- NULL until resolved, 1=YES, 0=NO
);

CREATE TABLE portfolio (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Use WAL mode: `PRAGMA journal_mode=WAL;`

### News Pipeline Priority
1. **pygooglenews** — free, no API key, Google News RSS
2. **gdeltdoc** — free, GDELT event database, 15-min updates
3. **newsapi-python** — 500 free requests/day, 150K+ sources
4. Cache all results for 30 minutes per market

## Code Style

- Python 3.11+, async/await with `asyncio`
- Type hints on all function signatures
- Pydantic models for all data structures
- `structlog` for structured JSON logging
- `ruff` for linting and formatting
- No print statements — use logger
- All external API calls: try/except with exponential backoff
- Every function touching external APIs gets `@observe`

## Testing

- Unit tests mock all external APIs (Kalshi, Claude, news)
- **Demo environment is the integration test** — run full loop against `demo-api.kalshi.co`
- Test risk manager edge cases: drawdown triggers, limit enforcement
- Test forecaster: mock Claude, verify ensemble + extremization math
- Test edge detection: Kelly calculation, minimum edge filtering

## Common Pitfalls

- **Prices are CENTS (integers), not decimals** — 65 means $0.65. Convert to probability: `p = price / 100`. When placing orders, `yes_price` takes int 1-99.
- **Tickers, not token IDs** — Kalshi uses human-readable tickers like `FED-RATE-CUT-JUN26`, not opaque token IDs.
- **Events contain markets** — an event like "Fed rate decision" has multiple markets for different outcomes. Use `GET /events` to understand groupings.
- **Auth tokens expire every 30 minutes** — handle refresh in the client wrapper.
- **Rate limits return 429** — implement exponential backoff. Kalshi uses tiered rate limits.
- **Demo vs Prod are separate** — different API keys, different base URLs, separate account balances. Always test in demo first.
- **No WebSocket auth in demo** — WebSocket works differently in demo vs production.
- **Balance is in cents** — `balance.balance` returns cents. Divide by 100 for dollars.
- **Extremization is critical** — raw LLM probabilities hedge toward 50%.

## Environment

- Python 3.11+
- Proxmox LXC: 1 CPU, 1GB RAM, 5GB disk
- systemd service for auto-restart
- RSA private key PEM file with 600 permissions
- `.env` file with 600 permissions