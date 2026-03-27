"""Claude forecaster — cheap Haiku screen + Sonnet ensemble on promising markets."""

import datetime

import anthropic
import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.edge import aggregate_forecasts
from kalshiedge.prompts import FORECAST_USER, SUPERFORECASTER_SYSTEM, parse_forecast
from kalshiedge.research import NewsItem, format_news_context

logger = structlog.get_logger()

SCREEN_MODEL = "claude-haiku-4-5-20251001"
SCREEN_MAX_TOKENS = 256

SCREEN_PROMPT = (
    "You are a quick probability screener. Given a prediction market question, "
    "current price, and news, estimate the probability in one line.\n"
    "Respond ONLY with: PROBABILITY: <number>%"
)


class ForecastResult:
    def __init__(
        self,
        probability: float,
        confidence_low: float | None,
        confidence_high: float | None,
        reasoning: str | None,
        raw_probabilities: list[float],
    ):
        self.probability = probability
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.reasoning = reasoning
        self.raw_probabilities = raw_probabilities


async def screen_market(
    client: anthropic.Anthropic,
    title: str,
    price_cents: int,
    close_time: str,
    news_items: list[NewsItem],
) -> float | None:
    """Quick Haiku screen — returns estimated probability or None on failure.

    Cost: ~$0.001 per call (vs ~$0.017 for full Sonnet ensemble).
    """
    news_context = format_news_context(news_items)
    price_pct = price_cents / 100
    prompt = (
        f"QUESTION: {title}\n"
        f"MARKET PRICE: {price_cents}c ({price_pct:.0%})\n"
        f"CLOSES: {close_time}\n"
        f"NEWS: {news_context[:500]}\n\n"
        f"PROBABILITY: "
    )
    try:
        response = client.messages.create(
            model=SCREEN_MODEL,
            max_tokens=SCREEN_MAX_TOKENS,
            system=SCREEN_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        parsed = parse_forecast(f"PROBABILITY: {text}")
        if parsed["probability"] is not None:
            logger.debug(
                "screen_result",
                title=title[:50],
                prob=f"{parsed['probability']:.0%}",
                price=price_cents,
            )
            return parsed["probability"]
    except Exception:
        logger.debug("screen_failed", title=title[:50])
    return None


@observe(span_type=SpanType.CHAIN)
async def forecast_market(
    client: anthropic.Anthropic,
    title: str,
    price_cents: int,
    close_time: str,
    news_items: list[NewsItem],
) -> ForecastResult | None:
    """Full Sonnet ensemble — 3 calls at different temperatures."""
    news_context = format_news_context(news_items)
    current_date = datetime.date.today().isoformat()

    # Anti-anchoring: don't show market price to the model
    user_prompt = FORECAST_USER.format(
        title=title,
        close_time=close_time,
        current_date=current_date,
        news_context=news_context,
    )

    raw_probabilities: list[float] = []
    last_parsed: dict | None = None

    for temp in settings.temperatures:
        try:
            response = client.messages.create(
                model=settings.forecast_model,
                max_tokens=1024,
                temperature=temp,
                system=SUPERFORECASTER_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text
            parsed = parse_forecast(text)
            if parsed["probability"] is not None:
                raw_probabilities.append(parsed["probability"])
                last_parsed = parsed
            else:
                logger.warning("forecast_parse_failed", temperature=temp, title=title)
        except Exception:
            logger.exception("forecast_call_failed", temperature=temp, title=title)

    if not raw_probabilities:
        logger.error("all_forecasts_failed", title=title)
        return None

    aggregated = aggregate_forecasts(raw_probabilities)

    logger.info(
        "forecast_complete",
        title=title,
        raw=raw_probabilities,
        aggregated=aggregated,
        price_cents=price_cents,
    )

    return ForecastResult(
        probability=aggregated,
        confidence_low=last_parsed.get("confidence_low") if last_parsed else None,
        confidence_high=last_parsed.get("confidence_high") if last_parsed else None,
        reasoning=last_parsed.get("reasoning") if last_parsed else None,
        raw_probabilities=raw_probabilities,
    )
