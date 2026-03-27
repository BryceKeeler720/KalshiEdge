# SKILLS.md — KalshiEdge Domain Reference

Domain-specific technical knowledge for building the KalshiEdge autonomous Kalshi trading agent.

---

## 1. Kalshi API Reference

### Base URLs
```
Production REST:  https://trading-api.kalshi.com/trade-api/v2
Production WS:    wss://trading-api.kalshi.com/trade-api/ws/v2
Demo REST:        https://demo-api.kalshi.co/trade-api/v2
Demo WS:          wss://demo-api.kalshi.co/trade-api/ws/v2
```

### Authentication (RSA-PSS)

Generate an API key in the Kalshi dashboard → download the private key PEM file.

Every authenticated request includes three headers:
- `KALSHI-ACCESS-KEY`: your API key ID
- `KALSHI-ACCESS-SIGNATURE`: RSA-PSS signature of `timestamp + method + path`
- `KALSHI-ACCESS-TIMESTAMP`: current Unix timestamp (ISO 8601)

```python
import base64, time, datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import httpx

class KalshiAuth:
    def __init__(self, api_key_id: str, private_key_pem: str, base_url: str):
        self.api_key_id = api_key_id
        self.base_url = base_url
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )

    def _sign(self, timestamp: str, method: str, path: str) -> str:
        message = f"{timestamp}{method}{path}".encode()
        sig = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def headers(self, method: str, path: str) -> dict:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }
```

### Using the Official SDK (`kalshi-python`)
```python
from kalshi_python import Configuration, KalshiClient

config = Configuration(host="https://demo-api.kalshi.co/trade-api/v2")
with open("private_key.pem", "r") as f:
    config.private_key_pem = f.read()
config.api_key_id = "your-api-key-id"

client = KalshiClient(config)
balance = client.get_balance()
print(f"Balance: ${balance.balance / 100:.2f}")
```

### Market Data (Public, No Auth)

**List markets:**
```
GET /markets?status=open&limit=100&cursor=CURSOR
GET /markets?series_ticker=FED-RATE&status=open
GET /markets/{ticker}
GET /markets/{ticker}/orderbook
GET /markets/{ticker}/trades
```

**List events (groups of related markets):**
```
GET /events?status=open&limit=50
GET /events/{event_ticker}
```

**Key market response fields:**
```json
{
  "ticker": "FED-RATE-CUT-JUN26",
  "event_ticker": "FED-RATE-JUN26",
  "title": "Will the Fed cut rates in June 2026?",
  "status": "open",
  "yes_bid": 62,
  "yes_ask": 65,
  "no_bid": 35,
  "no_ask": 38,
  "last_price": 63,
  "volume": 15420,
  "volume_24h": 2340,
  "open_interest": 8750,
  "close_time": "2026-06-18T18:00:00Z",
  "expiration_time": "2026-06-18T18:00:00Z",
  "category": "Economics",
  "series_ticker": "FED-RATE"
}
```

- **Prices are in CENTS (integers 1-99)** — `yes_bid: 62` means 62¢ = 62% implied probability
- `yes_bid` / `yes_ask` = best bid/ask for YES side
- `no_bid` / `no_ask` = best bid/ask for NO side
- `volume_24h` = 24-hour volume in contracts
- `open_interest` = total outstanding contracts
- `close_time` = when trading stops
- Use cursor-based pagination for large result sets

### Order Management (Authenticated)

**Place order:**
```
POST /portfolio/orders
{
  "ticker": "FED-RATE-CUT-JUN26",
  "action": "buy",           // "buy" or "sell"
  "side": "yes",             // "yes" or "no"
  "type": "limit",           // "limit" or "market"
  "count": 10,               // Number of contracts
  "yes_price": 55            // Price in cents (limit orders only)
}
```

**Get orders:**
```
GET /portfolio/orders?ticker=FED-RATE-CUT-JUN26&status=resting
```

**Cancel order:**
```
DELETE /portfolio/orders/{order_id}
```

**Order statuses:** `resting` (on book), `pending` (being processed), `executed` (filled), `canceled`

### Portfolio (Authenticated)

```
GET /portfolio/balance          → { "balance": 10000 }  // cents
GET /portfolio/positions        → list of open positions
GET /portfolio/settlements      → settlement history
```

### WebSocket

```python
import websockets, json

async def stream_orderbook(ticker: str):
    uri = "wss://trading-api.kalshi.com/trade-api/ws/v2"
    async with websockets.connect(uri) as ws:
        # Authenticate first
        await ws.send(json.dumps({
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [ticker]
            }
        }))
        async for msg in ws:
            data = json.loads(msg)
            print(data)
```

Channels: `orderbook_delta`, `ticker`, `trade`, `fill` (authenticated)

### Rate Limits

Kalshi uses tiered rate limits. Default tier returns 429 on exceeded. Implement exponential backoff. Key limits:
- Market data endpoints: generous (hundreds/min)
- Order placement: more restrictive
- Tier upgrades available via Builder Program (contact Kalshi)
- Cache market data aggressively to reduce calls

---

## 2. Kelly Criterion for Binary Markets

### Formula
```
f* = (b * p - q) / b

where:
  p = estimated true probability of YES
  q = 1 - p
  b = (100 - market_price_cents) / market_price_cents  (odds)
  f* = fraction of bankroll to bet
```

### Quarter Kelly Implementation
```python
def quarter_kelly(p_model: float, price_cents: int, bankroll_cents: int) -> int:
    """Calculate quarter-Kelly position size in cents.

    Args:
        p_model: estimated true probability (0.0-1.0)
        price_cents: Kalshi market price (1-99)
        bankroll_cents: current bankroll in cents

    Returns:
        Position size in cents (cost of contracts)
    """
    p_market = price_cents / 100
    if p_model <= p_market:
        return 0

    b = (100 - price_cents) / price_cents
    q = 1 - p_model
    full_kelly = (b * p_model - q) / b

    if full_kelly <= 0:
        return 0

    quarter = full_kelly * 0.25
    max_size = int(bankroll_cents * 0.05)  # 5% cap
    position_cents = min(int(quarter * bankroll_cents), max_size)

    # Convert to whole number of contracts
    num_contracts = position_cents // price_cents
    return num_contracts * price_cents  # Actual cost


def contracts_from_budget(budget_cents: int, price_cents: int) -> int:
    """How many contracts can we buy with this budget."""
    return budget_cents // price_cents
```

### For NO Side
```python
def kelly_no(p_model: float, price_cents: int, bankroll_cents: int) -> int:
    """Kelly for buying NO."""
    no_price = 100 - price_cents
    p_no = 1 - p_model
    return quarter_kelly(p_no, no_price, bankroll_cents)
```

### Edge Calculation
```python
def compute_edge(p_model: float, price_cents: int) -> tuple[str, float]:
    """Returns (side, edge_magnitude)."""
    p_market = price_cents / 100
    edge_yes = p_model - p_market
    edge_no = p_market - p_model

    if edge_yes > edge_no and edge_yes > 0:
        return "yes", edge_yes
    elif edge_no > 0:
        return "no", edge_no
    return "none", 0.0
```

---

## 3. Extremization (LLM Calibration Correction)

```python
import math

def extremize(p: float, factor: float = 1.3) -> float:
    """Push probability away from 0.5 by multiplying log-odds."""
    if p <= 0.01 or p >= 0.99:
        return p
    log_odds = math.log(p / (1 - p))
    adjusted = log_odds * factor
    return 1 / (1 + math.exp(-adjusted))


def aggregate_forecasts(probabilities: list[float]) -> float:
    """Trimmed mean: drop highest and lowest, average rest, then extremize."""
    if len(probabilities) <= 2:
        raw = sum(probabilities) / len(probabilities)
    else:
        sorted_p = sorted(probabilities)
        trimmed = sorted_p[1:-1]
        raw = sum(trimmed) / len(trimmed)
    return extremize(raw)
```

---

## 4. Forecaster Prompt Template

```python
SUPERFORECASTER_SYSTEM = """You are an expert forecaster trained to minimize Brier scores. \
You reason carefully about base rates, reference classes, and evidence quality. \
You provide calibrated probability estimates — when you say 70%, events should occur ~70% of the time.

Rules:
1. Start with the base rate for this class of event.
2. Adjust based on specific evidence, noting direction and magnitude.
3. Explicitly state what evidence would change your estimate.
4. Do NOT hedge toward 50% — be as precise as your evidence supports.
5. Respond with the exact format specified."""

FORECAST_USER = """QUESTION: {title}
CURRENT MARKET PRICE: {price_cents}¢ ({price_pct:.0%} implied probability)
MARKET CLOSES: {close_time}
TODAY: {current_date}

RECENT NEWS AND CONTEXT:
{news_context}

Analyze this market and provide your probability estimate.

Respond in EXACTLY this format:
PROBABILITY: <number>%
CONFIDENCE_RANGE: [<low>%, <high>%]
KEY_FACTORS:
1. <factor 1>
2. <factor 2>
3. <factor 3>
REASONING: <2-3 sentence summary>"""
```

### Parsing
```python
import re

def parse_forecast(text: str) -> dict:
    prob = re.search(r"PROBABILITY:\s*([\d.]+)%", text)
    rng = re.search(r"CONFIDENCE_RANGE:\s*\[([\d.]+)%,\s*([\d.]+)%\]", text)
    reasoning = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    return {
        "probability": float(prob.group(1)) / 100 if prob else None,
        "confidence_low": float(rng.group(1)) / 100 if rng else None,
        "confidence_high": float(rng.group(2)) / 100 if rng else None,
        "reasoning": reasoning.group(1).strip() if reasoning else None,
    }
```

---

## 5. News Gathering

### Google News RSS (Free)
```python
from pygooglenews import GoogleNews

gn = GoogleNews(lang="en", country="US")

def search_news(query: str, days: int = 7) -> list[dict]:
    results = gn.search(query, when=f"{days}d")
    return [
        {"title": e["title"], "published": e["published"],
         "source": e.get("source", {}).get("title")}
        for e in results.get("entries", [])[:5]
    ]
```

### GDELT (Free, 15-min Updates)
```python
from gdeltdoc import GdeltDoc, Filters

gd = GdeltDoc()

def search_gdelt(query: str, days: int = 7) -> list[dict]:
    f = Filters(keyword=query, timespan=f"{days}d")
    articles = gd.article_search(f)
    return articles.head(5).to_dict("records") if not articles.empty else []
```

### NewsAPI (500 Free/Day)
```python
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

def search_newsapi(query: str) -> list[dict]:
    resp = newsapi.get_everything(q=query, sort_by="publishedAt", page_size=5, language="en")
    return [
        {"title": a["title"], "description": a["description"],
         "source": a["source"]["name"]}
        for a in resp.get("articles", [])
    ]
```

---

## 6. Fee Awareness

### Kalshi Fee Structure
```python
def estimated_taker_fee_cents(price_cents: int) -> float:
    """Approximate taker fee per contract in cents.
    Fees peak at 50¢ (~1.7¢) and drop toward extremes.
    Maker (resting limit) orders are FREE.
    """
    # Approximation: fee ≈ base * 4 * p * (1-p)
    p = price_cents / 100
    base_rate = 0.07  # ~7% of expected profit at peak
    fee = base_rate * 4 * p * (1 - p)
    return round(fee * 100, 2)  # Convert to cents

# At 50¢: ~1.7¢ per contract
# At 80¢: ~1.1¢ per contract
# At 95¢: ~0.3¢ per contract
```

### Edge Threshold with Fees
```python
def net_edge(p_model: float, price_cents: int) -> float:
    """Edge after accounting for worst-case taker exit fee."""
    side, raw_edge = compute_edge(p_model, price_cents)
    # Assume we might need to exit as taker
    exit_price = int(p_model * 100)
    fee_cents = estimated_taker_fee_cents(exit_price)
    spread_cents = 1  # Typical 1¢ spread
    cost_pct = (fee_cents + spread_cents) / 100
    return raw_edge - cost_pct
```

---

## 7. 2signal Span Mapping

```python
from twosignal import TwoSignal, observe
from twosignal.types import SpanType
from twosignal.wrappers.anthropic import wrap_anthropic
import anthropic

ts = TwoSignal(
    api_key=os.getenv("TWOSIGNAL_API_KEY"),
    base_url=os.getenv("TWOSIGNAL_BASE_URL"),
    deployment_id="kalshiedge-v0.1.0",
)
claude = wrap_anthropic(anthropic.Anthropic())

@observe(span_type=SpanType.AGENT, evaluators=["LATENCY"])
def run_cycle():
    markets = discover_markets()
    for m in markets:
        signal = forecast_and_decide(m)
        if signal and signal.edge > MIN_EDGE:
            execute_trade(m, signal)

@observe(span_type=SpanType.RETRIEVAL)
def discover_markets() -> list[Market]: ...

@observe(span_type=SpanType.RETRIEVAL)
def gather_news(market: Market) -> list[NewsItem]: ...

@observe(span_type=SpanType.CHAIN)
def forecast_and_decide(market: Market) -> TradeSignal | None:
    news = gather_news(market)
    forecasts = []
    for temp in [0.3, 0.5, 0.7]:
        result = run_single_forecast(market, news, temperature=temp)
        forecasts.append(result)
    aggregated = aggregate_forecasts(forecasts)
    return evaluate_edge(market, aggregated)

@observe(span_type=SpanType.TOOL)
def execute_trade(market: Market, signal: TradeSignal): ...

@observe(span_type=SpanType.GUARDRAIL)
def check_risk_limits(trade: ProposedTrade) -> bool: ...

import atexit
atexit.register(ts.shutdown)
```

---

## 8. Proxmox LXC Deployment

### systemd Service
```ini
# /etc/systemd/system/kalshiedge.service
[Unit]
Description=KalshiEdge Trading Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=kalshiedge
WorkingDirectory=/opt/kalshiedge
EnvironmentFile=/opt/kalshiedge/.env
ExecStart=/opt/kalshiedge/.venv/bin/python -m kalshiedge.main
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Setup
```bash
apt update && apt install -y python3.11 python3.11-venv git
useradd -m -s /bin/bash kalshiedge
mkdir -p /opt/kalshiedge && chown kalshiedge:kalshiedge /opt/kalshiedge
cd /opt/kalshiedge
python3.11 -m venv .venv && source .venv/bin/activate
git clone <repo> . && pip install -e .
cp .env.example .env && chmod 600 .env
# Copy private_key.pem and chmod 600
sudo systemctl enable kalshiedge && sudo systemctl start kalshiedge
```

---

## 9. Calibration Tracking

```python
def brier_score(forecasts: list[tuple[float, int]]) -> float:
    """(predicted_probability, actual_outcome) → lower is better."""
    if not forecasts:
        return float('nan')
    return sum((p - o) ** 2 for p, o in forecasts) / len(forecasts)

def calibration_table(forecasts: list[tuple[float, int]], bins: int = 10) -> dict:
    from collections import defaultdict
    buckets = defaultdict(list)
    for p, o in forecasts:
        bucket = min(int(p * bins), bins - 1)
        buckets[bucket].append(o)
    return {
        f"{b/bins:.0%}-{(b+1)/bins:.0%}": {
            "count": len(outcomes),
            "predicted_avg": (b + 0.5) / bins,
            "actual_freq": sum(outcomes) / len(outcomes) if outcomes else 0,
        }
        for b, outcomes in sorted(buckets.items())
    }
```

**Go-live criteria:** 50+ demo forecasts, Brier ≤ 0.15, positive simulated P&L, no systematic bias >5% in any calibration bucket.

---

## 10. Kalshi vs Polymarket Quick Comparison

| Feature | Kalshi | Polymarket |
|---------|--------|------------|
| Legal for US | Yes (CFTC regulated) | Invite-only waitlist, KYC required |
| Settlement | USD (ACH) | USDC.e on Polygon |
| API count | 1 unified API | 3 APIs (Gamma, CLOB, Data) |
| Auth | RSA-PSS | Blockchain wallet (EIP-712) |
| Prices | Cents (int 1-99) | Decimal (0.00-1.00) |
| Maker fees | Free | Free (+ rebates) |
| Demo env | Yes (demo-api.kalshi.co) | No |
| Heartbeat | Not needed | Required (10s or orders cancel) |
| Liquidity | Lower | Higher |
| Market count | Fewer | More |
| Identifiers | Tickers (FED-RATE-CUT-JUN26) | Token IDs (hex strings) |