"""Bull/bear debate protocol — adversarial forecasting at lower cost.

Instead of 3x Sonnet calls ($0.051), runs:
1. Haiku bull case ($0.001)
2. Haiku bear case ($0.001)
3. Sonnet judge ($0.017)
Total: $0.019 — 63% cheaper, arguably more accurate due to adversarial debiasing.
"""


import anthropic
import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.edge import extremize
from kalshiedge.prompts import (
    BEAR_PROMPT,
    BULL_PROMPT,
    JUDGE_PROMPT,
    parse_bull_bear,
    parse_forecast,
)
from kalshiedge.research import NewsItem, format_news_context

logger = structlog.get_logger()

HAIKU_MODEL = "claude-haiku-4-5-20251001"


class DebateResult:
    def __init__(
        self,
        probability: float,
        bull_prob: float | None,
        bear_prob: float | None,
        reasoning: str | None,
        disagreement: float,
    ):
        self.probability = probability
        self.bull_prob = bull_prob
        self.bear_prob = bear_prob
        self.reasoning = reasoning
        self.disagreement = disagreement
        self.confidence_low = None
        self.confidence_high = None
        self.raw_probabilities = [p for p in [bull_prob, bear_prob, probability] if p]


@observe(span_type=SpanType.CHAIN)
async def debate_forecast(
    client: anthropic.Anthropic,
    title: str,
    price_cents: int,
    close_time: str,
    news_items: list[NewsItem],
) -> DebateResult | None:
    """Run bull/bear debate: 2 Haiku + 1 Sonnet judge."""
    news_context = format_news_context(news_items)

    # Step 1: Bull case (Haiku — cheap)
    bull_prob = None
    bull_text = ""
    try:
        bull_resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": BULL_PROMPT.format(
                    title=title,
                    close_time=close_time,
                    news_context=news_context[:500],
                ),
            }],
        )
        bull_text = bull_resp.content[0].text
        bull_prob, _ = parse_bull_bear(bull_text)
    except Exception:
        logger.warning("debate_bull_failed", title=title[:50])

    # Step 2: Bear case (Haiku — cheap)
    bear_prob = None
    bear_text = ""
    try:
        bear_resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": BEAR_PROMPT.format(
                    title=title,
                    close_time=close_time,
                    bull_case=bull_text[:300],
                    news_context=news_context[:500],
                ),
            }],
        )
        bear_text = bear_resp.content[0].text
        bear_prob, _ = parse_bull_bear(bear_text)
    except Exception:
        logger.warning("debate_bear_failed", title=title[:50])

    # Step 3: Judge (Sonnet — one call instead of three)
    judge_prob = None
    reasoning = None
    try:
        judge_resp = client.messages.create(
            model=settings.forecast_model,
            max_tokens=512,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    title=title,
                    close_time=close_time,
                    bull_case=bull_text[:400],
                    bear_case=bear_text[:400],
                ),
            }],
        )
        judge_text = judge_resp.content[0].text
        parsed = parse_forecast(judge_text)
        judge_prob = parsed["probability"]
        reasoning = parsed["reasoning"]
    except Exception:
        logger.warning("debate_judge_failed", title=title[:50])

    # Aggregate: if judge failed, use average of bull/bear
    probs = [p for p in [bull_prob, bear_prob, judge_prob] if p is not None]
    if not probs:
        logger.error("debate_all_failed", title=title[:50])
        return None

    if judge_prob is not None:
        raw_prob = judge_prob
    else:
        raw_prob = sum(probs) / len(probs)

    # Apply extremization
    final_prob = extremize(raw_prob)

    # Measure disagreement between bull and bear
    disagreement = 0.0
    if bull_prob is not None and bear_prob is not None:
        disagreement = abs(bull_prob - bear_prob)

    # Penalize confidence when agents disagree heavily (>30%)
    if disagreement > 0.30:
        penalty = min(1.0, disagreement / 0.5) * 0.15
        final_prob = final_prob * (1 - penalty) + 0.5 * penalty

    logger.info(
        "debate_complete",
        title=title[:50],
        bull=f"{bull_prob:.0%}" if bull_prob else "N/A",
        bear=f"{bear_prob:.0%}" if bear_prob else "N/A",
        judge=f"{judge_prob:.0%}" if judge_prob else "N/A",
        final=f"{final_prob:.0%}",
        disagreement=f"{disagreement:.0%}",
    )

    return DebateResult(
        probability=final_prob,
        bull_prob=bull_prob,
        bear_prob=bear_prob,
        reasoning=reasoning,
        disagreement=disagreement,
    )
