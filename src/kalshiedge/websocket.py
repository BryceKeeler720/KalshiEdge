"""WebSocket client for real-time Kalshi orderbook and fill updates."""

import asyncio
import json

import structlog
import websockets

from kalshiedge.config import settings
from kalshiedge.discovery import _dollars_to_cents
from kalshiedge.kalshi_client import KalshiAuth

logger = structlog.get_logger()

# New API domain for WebSocket
WS_URL_PROD = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_URL_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"

RECONNECT_DELAY = 5
MAX_RECONNECT_DELAY = 60


class KalshiWebSocket:
    """Persistent WebSocket connection for real-time market data and fills."""

    def __init__(self) -> None:
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._subscriptions: set[str] = set()
        self._orderbook_cache: dict[str, dict] = {}  # ticker -> {yes_bid, yes_ask, ...}
        self._fill_callbacks: list = []
        self._running = False
        self._auth = KalshiAuth(settings.kalshi_api_key_id, settings.private_key_pem)

    @property
    def ws_url(self) -> str:
        if settings.kalshi_env == "prod":
            return WS_URL_PROD
        return WS_URL_DEMO

    def get_cached_price(self, ticker: str) -> dict | None:
        """Get cached orderbook data for a ticker."""
        return self._orderbook_cache.get(ticker)

    async def start(self) -> None:
        """Start the WebSocket connection in the background."""
        self._running = True
        asyncio.create_task(self._connection_loop())
        logger.info("websocket_started")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("websocket_stopped")

    async def subscribe_tickers(self, tickers: list[str]) -> None:
        """Subscribe to orderbook updates for given tickers."""
        new_tickers = [t for t in tickers if t not in self._subscriptions]
        if not new_tickers or not self._ws:
            return

        try:
            await self._ws.send(json.dumps({
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta", "ticker"],
                    "market_tickers": new_tickers,
                },
            }))
            self._subscriptions.update(new_tickers)
            logger.info("ws_subscribed", tickers=len(new_tickers))
        except Exception:
            logger.warning("ws_subscribe_failed")

    async def subscribe_fills(self) -> None:
        """Subscribe to fill notifications (authenticated)."""
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps({
                "id": 2,
                "cmd": "subscribe",
                "params": {"channels": ["fill"]},
            }))
            logger.info("ws_fill_subscription_active")
        except Exception:
            logger.warning("ws_fill_subscribe_failed")

    async def _connection_loop(self) -> None:
        """Maintain persistent connection with reconnection logic."""
        delay = RECONNECT_DELAY

        while self._running:
            try:
                # Auth headers for WebSocket
                auth_headers = self._auth.headers("GET", "/trade-api/ws/v2")

                async with websockets.connect(
                    self.ws_url,
                    additional_headers=auth_headers,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    delay = RECONNECT_DELAY
                    logger.info("ws_connected", url=self.ws_url)

                    # Re-subscribe after reconnect
                    if self._subscriptions:
                        await self.subscribe_tickers(list(self._subscriptions))
                    await self.subscribe_fills()

                    async for msg in ws:
                        await self._handle_message(msg)

            except websockets.ConnectionClosed:
                logger.warning("ws_disconnected", reconnect_in=delay)
            except Exception:
                logger.exception("ws_error", reconnect_in=delay)

            self._ws = None
            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)

    async def _handle_message(self, raw: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", "")
        msg = data.get("msg", data)

        if msg_type == "orderbook_delta" or "orderbook" in str(data.get("channel", "")):
            await self._handle_orderbook(msg)
        elif msg_type == "ticker" or "ticker" in str(data.get("channel", "")):
            await self._handle_ticker(msg)
        elif msg_type == "fill" or "fill" in str(data.get("channel", "")):
            await self._handle_fill(msg)

    async def _handle_orderbook(self, msg: dict) -> None:
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        # Update cache with new orderbook data
        if ticker not in self._orderbook_cache:
            self._orderbook_cache[ticker] = {}

        cache = self._orderbook_cache[ticker]
        if "yes_bid" in msg:
            cache["yes_bid"] = _dollars_to_cents(msg.get("yes_bid"))
        if "yes_ask" in msg:
            cache["yes_ask"] = _dollars_to_cents(msg.get("yes_ask"))
        if "no_bid" in msg:
            cache["no_bid"] = _dollars_to_cents(msg.get("no_bid"))
        if "no_ask" in msg:
            cache["no_ask"] = _dollars_to_cents(msg.get("no_ask"))

    async def _handle_ticker(self, msg: dict) -> None:
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        if ticker not in self._orderbook_cache:
            self._orderbook_cache[ticker] = {}

        cache = self._orderbook_cache[ticker]
        for field in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_price", "volume"):
            if field in msg:
                cache[field] = _dollars_to_cents(msg[field])

    async def _handle_fill(self, msg: dict) -> None:
        logger.info(
            "ws_fill_received",
            ticker=msg.get("market_ticker"),
            side=msg.get("side"),
            count=msg.get("count"),
            price=msg.get("yes_price"),
        )
        for callback in self._fill_callbacks:
            try:
                await callback(msg)
            except Exception:
                logger.warning("fill_callback_failed")

    def on_fill(self, callback) -> None:
        """Register a callback for fill events."""
        self._fill_callbacks.append(callback)
