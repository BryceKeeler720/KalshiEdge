"""Superforecaster prompt templates and response parsing."""

import re

SUPERFORECASTER_SYSTEM = (
    "You are an expert forecaster trained to minimize Brier scores. "
    "You reason carefully about base rates, reference classes, and evidence quality. "
    "You provide calibrated probability estimates — when you say 70%, "
    "events should occur ~70% of the time.\n\n"
    "Rules:\n"
    "1. Start with the base rate for this class of event.\n"
    "2. Adjust based on specific evidence, noting direction and magnitude.\n"
    "3. Explicitly state what evidence would change your estimate.\n"
    "4. Do NOT hedge toward 50% — be as precise as your evidence supports.\n"
    "5. Respond with the exact format specified."
)

FORECAST_USER = (
    "QUESTION: {title}\n"
    "CURRENT MARKET PRICE: {price_cents}\u00a2 ({price_pct:.0%} implied probability)\n"
    "MARKET CLOSES: {close_time}\n"
    "TODAY: {current_date}\n\n"
    "RECENT NEWS AND CONTEXT:\n"
    "{news_context}\n\n"
    "Analyze this market and provide your probability estimate.\n\n"
    "Respond in EXACTLY this format:\n"
    "PROBABILITY: <number>%\n"
    "CONFIDENCE_RANGE: [<low>%, <high>%]\n"
    "KEY_FACTORS:\n"
    "1. <factor 1>\n"
    "2. <factor 2>\n"
    "3. <factor 3>\n"
    "REASONING: <2-3 sentence summary>"
)


def parse_forecast(text: str) -> dict:
    """Parse structured forecast response from Claude."""
    prob = re.search(r"PROBABILITY:\s*([\d.]+)%", text)
    rng = re.search(r"CONFIDENCE_RANGE:\s*\[([\d.]+)%,\s*([\d.]+)%\]", text)
    reasoning = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    return {
        "probability": float(prob.group(1)) / 100 if prob else None,
        "confidence_low": float(rng.group(1)) / 100 if rng else None,
        "confidence_high": float(rng.group(2)) / 100 if rng else None,
        "reasoning": reasoning.group(1).strip() if reasoning else None,
    }
