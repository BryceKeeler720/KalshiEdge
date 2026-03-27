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
    "5. Do NOT anchor to any market price — estimate independently.\n"
    "6. Respond with the exact format specified."
)

# Anti-anchoring: no market price shown to the model
FORECAST_USER = (
    "QUESTION: {title}\n"
    "MARKET CLOSES: {close_time}\n"
    "TODAY: {current_date}\n\n"
    "RECENT NEWS AND CONTEXT:\n"
    "{news_context}\n\n"
    "Analyze this question and provide your independent probability estimate.\n\n"
    "Respond in EXACTLY this format:\n"
    "PROBABILITY: <number>%\n"
    "CONFIDENCE_RANGE: [<low>%, <high>%]\n"
    "KEY_FACTORS:\n"
    "1. <factor 1>\n"
    "2. <factor 2>\n"
    "3. <factor 3>\n"
    "REASONING: <2-3 sentence summary>"
)

# Debate prompts for bull/bear protocol
BULL_PROMPT = (
    "QUESTION: {title}\n"
    "CLOSES: {close_time}\n"
    "NEWS: {news_context}\n\n"
    "Make the STRONGEST case for YES. What evidence supports this happening? "
    "What base rates and precedents favor YES?\n"
    "End with: BULL_PROBABILITY: <number>%"
)

BEAR_PROMPT = (
    "QUESTION: {title}\n"
    "CLOSES: {close_time}\n"
    "BULL CASE: {bull_case}\n"
    "NEWS: {news_context}\n\n"
    "Make the STRONGEST case for NO. Challenge every bull argument. "
    "What evidence and base rates favor NO?\n"
    "End with: BEAR_PROBABILITY: <number>%"
)

JUDGE_PROMPT = (
    "QUESTION: {title}\n"
    "CLOSES: {close_time}\n\n"
    "BULL CASE (argues YES):\n{bull_case}\n\n"
    "BEAR CASE (argues NO):\n{bear_case}\n\n"
    "You are the final judge. Weigh both arguments objectively. "
    "Which side has stronger evidence? Give your calibrated probability.\n\n"
    "PROBABILITY: <number>%\n"
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


def parse_bull_bear(text: str) -> tuple[float | None, str]:
    """Parse bull or bear probability from debate response."""
    prob = re.search(r"(?:BULL_PROBABILITY|BEAR_PROBABILITY):\s*([\d.]+)%", text)
    probability = float(prob.group(1)) / 100 if prob else None
    return probability, text.strip()
