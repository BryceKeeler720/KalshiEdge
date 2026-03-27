"""Brier score tracking and calibration analysis."""

from collections import defaultdict

import structlog

logger = structlog.get_logger()


def brier_score(forecasts: list[tuple[float, int]]) -> float:
    """Compute Brier score from (predicted_probability, actual_outcome) pairs.

    Lower is better. Perfect = 0.0, random = 0.25.
    """
    if not forecasts:
        return float("nan")
    return sum((p - o) ** 2 for p, o in forecasts) / len(forecasts)


def calibration_table(forecasts: list[tuple[float, int]], bins: int = 10) -> dict:
    """Group forecasts into probability buckets and compare predicted vs actual."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for p, o in forecasts:
        bucket = min(int(p * bins), bins - 1)
        buckets[bucket].append(o)
    return {
        f"{b / bins:.0%}-{(b + 1) / bins:.0%}": {
            "count": len(outcomes),
            "predicted_avg": (b + 0.5) / bins,
            "actual_freq": sum(outcomes) / len(outcomes) if outcomes else 0,
        }
        for b, outcomes in sorted(buckets.items())
    }


async def generate_report(db) -> str:
    """Generate calibration report from resolved forecasts in the database."""
    rows = await db.execute_fetchall(
        "SELECT model_probability, actual_outcome FROM forecasts WHERE actual_outcome IS NOT NULL"
    )
    if not rows:
        return "No resolved forecasts yet."

    pairs = [(row[0], row[1]) for row in rows]
    score = brier_score(pairs)
    table = calibration_table(pairs)

    lines = [
        f"Calibration Report ({len(pairs)} resolved forecasts)",
        f"Brier Score: {score:.4f}",
        "",
        "Bucket           | Count | Predicted | Actual",
        "-----------------|-------|-----------|-------",
    ]
    for bucket_name, data in table.items():
        lines.append(
            f"{bucket_name:17s}| {data['count']:5d} | "
            f"{data['predicted_avg']:9.1%} | {data['actual_freq']:.1%}"
        )

    report = "\n".join(lines)
    logger.info("calibration_report_generated", brier_score=score, num_forecasts=len(pairs))
    return report
