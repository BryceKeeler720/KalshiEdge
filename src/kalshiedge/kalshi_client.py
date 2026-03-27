"""Kalshi REST API client with RSA-PSS authentication and rate limit handling."""

import asyncio
import base64
import datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from pydantic import BaseModel

from kalshiedge.config import settings

logger = structlog.get_logger()

MAX_RETRIES = 5
BACKOFF_BASE = 1.0


class KalshiAuth:
    """RSA-PSS request signer for Kalshi API."""

    def __init__(self, api_key_id: str, private_key_pem: str):
        self.api_key_id = api_key_id
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )

    def _sign(self, timestamp: str, method: str, path: str) -> str:
        # Strip query params — only sign the path portion
        sign_path = path.split("?")[0]
        message = f"{timestamp}{method}{sign_path}".encode()
        sig = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def headers(self, method: str, full_path: str) -> dict[str, str]:
        ts = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, full_path),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }


class OrderResponse(BaseModel):
    order_id: str
    ticker: str
    status: str
    action: str
    side: str
    type: str
    yes_price: int
    count: int


class KalshiClient:
    """Async Kalshi REST API client with exponential backoff on 429s."""

    def __init__(self, base_url: str | None = None, auth: KalshiAuth | None = None):
        self.base_url = (base_url or settings.kalshi_base_url).rstrip("/")
        self.auth = auth
        self._client = httpx.AsyncClient(timeout=30.0)

    async def _ensure_auth(self) -> None:
        if self.auth is None:
            self.auth = KalshiAuth(settings.kalshi_api_key_id, settings.private_key_pem)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        authenticated: bool = False,
        json: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        # Full path for signing includes the /trade-api/v2 prefix
        full_path = urlparse(url).path

        for attempt in range(MAX_RETRIES):
            headers: dict[str, str] = {}
            if authenticated:
                await self._ensure_auth()
                headers = self.auth.headers(method.upper(), full_path)

            try:
                resp = await self._client.request(
                    method, url, headers=headers, json=json, params=params
                )

                if resp.status_code == 429:
                    wait = BACKOFF_BASE * (2**attempt)
                    logger.warning("rate_limited", attempt=attempt, wait_seconds=wait, path=path)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "api_error",
                    status=exc.response.status_code,
                    path=path,
                    body=exc.response.text[:500],
                )
                if exc.response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_BASE * (2**attempt))
                    continue
                raise
            except httpx.RequestError as exc:
                logger.error("request_error", path=path, error=str(exc))
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_BASE * (2**attempt))
                    continue
                raise

        raise RuntimeError(f"Max retries exceeded for {method} {path}")

    # --- Public market data endpoints ---

    async def get_markets(
        self,
        limit: int = 100,
        cursor: str | None = None,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        return await self._request("GET", "/markets", params=params)

    async def get_market(self, ticker: str) -> dict:
        return await self._request("GET", f"/markets/{ticker}")

    async def get_orderbook(self, ticker: str) -> dict:
        return await self._request("GET", f"/markets/{ticker}/orderbook")

    async def get_events(self, status: str = "open") -> dict:
        return await self._request("GET", "/events", params={"status": status})

    # --- Authenticated portfolio endpoints ---

    async def get_balance(self) -> dict:
        return await self._request("GET", "/portfolio/balance", authenticated=True)

    async def get_positions(self) -> dict:
        return await self._request("GET", "/portfolio/positions", authenticated=True)

    async def get_orders(
        self, ticker: str | None = None, status: str | None = None
    ) -> dict:
        params: dict[str, str] = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return await self._request("GET", "/portfolio/orders", authenticated=True, params=params)

    # --- Authenticated order management ---

    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        type: str,
        count: int,
        yes_price: int | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "type": type,
            "count": count,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        return await self._request(
            "POST", "/portfolio/orders", authenticated=True, json=body
        )

    async def cancel_order(self, order_id: str) -> dict:
        return await self._request(
            "DELETE", f"/portfolio/orders/{order_id}", authenticated=True
        )

    async def close(self) -> None:
        await self._client.aclose()
