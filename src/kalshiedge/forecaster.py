"""Claude ensemble forecaster with extremization."""

import datetime

import anthropic
import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.edge import aggregate_forecasts
from kalshiedge.prompts import FORECAST_USER, SUPERFORECASTER_SYSTEM, parse_forecast
from kalshiedge.research import NewsItem, format_news_context

logger = structlog.get_logger()

TEMPERATURES = [0.3, 0.5, 0.7]
FORECAST_MODEL = "claude-sonnet-4-6"


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


@observe(span_type=SpanType.CHAIN)
async def forecast_market(
    client: anthropic.Anthropic,
    title: str,
    price_cents: int,
    close_time: str,
    news_items: list[NewsItem],
) -> ForecastResult | None:
    """Run 3 Claude calls at different temperatures, aggregate via trimmed mean + extremization."""
    news_context = format_news_context(news_items)
    price_pct = price_cents / 100
    current_date = datetime.date.today().isoformat()

    user_prompt = FORECAST_USER.format(
        title=title,
        price_cents=price_cents,
        price_pct=price_pct,
        close_time=close_time,
        current_date=current_date,
        news_context=news_context,
    )

    raw_probabilities: list[float] = []
    last_parsed: dict | None = None

    for temp in TEMPERATURES:
        try:
            response = client.messages.create(
                model=FORECAST_MODEL,
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
