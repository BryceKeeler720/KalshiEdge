"""Batch forecaster — submit forecasts via Anthropic Batch API at 50% cost.

Submit/collect pattern:
- Cycle A: discover markets, submit batch → store batch_id
- Cycle B: check if batch done, collect results, evaluate edge, trade
"""

import datetime

import anthropic
import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.edge import aggregate_forecasts
from kalshiedge.prompts import FORECAST_USER, SUPERFORECASTER_SYSTEM, parse_forecast
from kalshiedge.research import format_news_context

logger = structlog.get_logger()


class BatchForecaster:
    """Manages async batch forecast submissions and result collection."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self._pending_batch_id: str | None = None
        self._pending_markets: dict[str, dict] = {}  # custom_id -> market info

    @property
    def has_pending_batch(self) -> bool:
        return self._pending_batch_id is not None

    @observe(span_type=SpanType.CHAIN)
    def submit_batch(
        self,
        markets: list[dict],
    ) -> str | None:
        """Submit a batch of forecast requests. Returns batch_id.

        Each market dict needs: ticker, title, price_cents, close_time, news_items
        """
        requests = []
        self._pending_markets = {}

        for market in markets:
            news_context = format_news_context(market["news_items"])
            current_date = datetime.date.today().isoformat()

            user_prompt = FORECAST_USER.format(
                title=market["title"],
                close_time=market["close_time"],
                current_date=current_date,
                news_context=news_context,
            )

            for i, temp in enumerate(settings.temperatures):
                custom_id = f"{market['ticker']}__temp{i}"
                # Use cache_control on system prompt for cost savings
                system_block = [
                    {
                        "type": "text",
                        "text": SUPERFORECASTER_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ] if settings.prompt_caching else SUPERFORECASTER_SYSTEM

                requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": settings.forecast_model,
                        "max_tokens": 1024,
                        "temperature": temp,
                        "system": system_block,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                })
                self._pending_markets[custom_id] = {
                    "ticker": market["ticker"],
                    "title": market["title"],
                    "price_cents": market["price_cents"],
                    "temp_index": i,
                }

        if not requests:
            return None

        try:
            batch = self.client.messages.batches.create(requests=requests)
            self._pending_batch_id = batch.id
            logger.info(
                "batch_submitted",
                batch_id=batch.id,
                num_requests=len(requests),
                num_markets=len(markets),
            )
            return batch.id
        except Exception:
            logger.exception("batch_submit_failed")
            return None

    @observe(span_type=SpanType.CHAIN)
    def collect_results(self) -> dict[str, dict] | None:
        """Check if batch is done and collect results. Returns {ticker: ForecastResult} or None.

        Returns None if batch is still processing.
        """
        if not self._pending_batch_id:
            return None

        try:
            batch = self.client.messages.batches.retrieve(self._pending_batch_id)
        except Exception:
            logger.warning("batch_status_check_failed", batch_id=self._pending_batch_id)
            return None

        if batch.processing_status != "ended":
            logger.info(
                "batch_still_processing",
                batch_id=self._pending_batch_id,
                status=batch.processing_status,
            )
            return None

        # Batch complete — collect results
        results_by_ticker: dict[str, list[float]] = {}
        reasoning_by_ticker: dict[str, str] = {}
        confidence_by_ticker: dict[str, dict] = {}

        try:
            for result in self.client.messages.batches.results(self._pending_batch_id):
                custom_id = result.custom_id
                market_info = self._pending_markets.get(custom_id)
                if not market_info:
                    continue

                ticker = market_info["ticker"]

                if result.result.type == "succeeded":
                    text = result.result.message.content[0].text
                    parsed = parse_forecast(text)
                    if parsed["probability"] is not None:
                        if ticker not in results_by_ticker:
                            results_by_ticker[ticker] = []
                        results_by_ticker[ticker].append(parsed["probability"])

                        if parsed["reasoning"]:
                            reasoning_by_ticker[ticker] = parsed["reasoning"]
                        if parsed["confidence_low"] is not None:
                            confidence_by_ticker[ticker] = {
                                "low": parsed["confidence_low"],
                                "high": parsed["confidence_high"],
                            }
                else:
                    logger.warning(
                        "batch_result_failed",
                        custom_id=custom_id,
                        error_type=result.result.type,
                    )
        except Exception:
            logger.exception("batch_results_collection_failed")
            self._pending_batch_id = None
            self._pending_markets = {}
            return None

        # Aggregate results per ticker
        forecasts = {}
        for ticker, probs in results_by_ticker.items():
            aggregated = aggregate_forecasts(probs)
            market_info = next(
                (v for v in self._pending_markets.values() if v["ticker"] == ticker),
                None,
            )
            conf = confidence_by_ticker.get(ticker, {})
            forecasts[ticker] = {
                "probability": aggregated,
                "raw_probabilities": probs,
                "reasoning": reasoning_by_ticker.get(ticker),
                "confidence_low": conf.get("low"),
                "confidence_high": conf.get("high"),
                "price_cents": market_info["price_cents"] if market_info else 0,
                "title": market_info["title"] if market_info else "",
            }

        logger.info(
            "batch_results_collected",
            batch_id=self._pending_batch_id,
            forecasts=len(forecasts),
        )

        # Clear pending state
        self._pending_batch_id = None
        self._pending_markets = {}
        return forecasts
