"""Backtest framework — replay historical forecasts to tune parameters.

Usage:
    python scripts/backtest.py [--db kalshiedge.db] [--extremize 1.0,1.1,1.2,1.3,1.5]
"""

import argparse
import math
import sqlite3

from kalshiedge.calibration import brier_score, calibration_table
from kalshiedge.edge import compute_edge, net_edge


def extremize(p: float, factor: float) -> float:
    if p <= 0.01 or p >= 0.99:
        return p
    log_odds = math.log(p / (1 - p))
    adjusted = log_odds * factor
    return 1 / (1 + math.exp(-adjusted))


def run_backtest(db_path: str, extremization_factors: list[float]) -> None:
    conn = sqlite3.connect(db_path)

    # Get resolved forecasts
    rows = conn.execute(
        """SELECT ticker, market_price_cents, model_probability, edge,
                  actual_outcome, strategy
           FROM forecasts WHERE actual_outcome IS NOT NULL"""
    ).fetchall()

    if not rows:
        print("No resolved forecasts to backtest. Let the bot run and markets settle.")
        return

    print(f"Backtesting {len(rows)} resolved forecasts\n")

    # Get trades for P&L simulation
    trades = conn.execute(
        """SELECT ticker, side, price_cents, count, pnl_cents, strategy
           FROM trades WHERE pnl_cents IS NOT NULL"""
    ).fetchall()
    conn.close()

    # Test different extremization factors
    print("=" * 70)
    print("EXTREMIZATION FACTOR SWEEP")
    print("=" * 70)

    best_brier = float("inf")
    best_factor = 1.0

    for factor in extremization_factors:
        pairs = []
        for row in rows:
            raw_prob = row[2]
            outcome = row[4]
            adjusted = extremize(raw_prob, factor)
            pairs.append((adjusted, outcome))

        score = brier_score(pairs)
        marker = ""
        if score < best_brier:
            best_brier = score
            best_factor = factor
            marker = " <-- BEST"

        print(f"  Factor {factor:.1f}: Brier = {score:.4f}{marker}")

    print(f"\nOptimal extremization factor: {best_factor:.1f} (Brier: {best_brier:.4f})")

    # Edge threshold sweep
    print("\n" + "=" * 70)
    print("EDGE THRESHOLD SWEEP")
    print("=" * 70)

    for threshold in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        would_trade = 0
        correct = 0
        for row in rows:
            price = row[1]
            prob = extremize(row[2], best_factor)
            outcome = row[4]
            side, edge = compute_edge(prob, price)
            eff = net_edge(prob, price)

            if side != "none" and eff >= threshold:
                would_trade += 1
                # Check if our side was correct
                if side == "yes" and outcome == 1:
                    correct += 1
                elif side == "no" and outcome == 0:
                    correct += 1

        win_rate = correct / would_trade if would_trade > 0 else 0
        print(f"  Threshold {threshold:.0%}: {would_trade} trades, {win_rate:.0%} win rate")

    # Calibration table
    print("\n" + "=" * 70)
    print("CALIBRATION TABLE")
    print("=" * 70)

    pairs = [(extremize(r[2], best_factor), r[4]) for r in rows]
    table = calibration_table(pairs)
    print(f"  {'Bucket':<15} {'Count':>6} {'Predicted':>10} {'Actual':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*8}")
    for bucket, data in table.items():
        print(
            f"  {bucket:<15} {data['count']:>6} "
            f"{data['predicted_avg']:>9.1%} {data['actual_freq']:>7.1%}"
        )

    # Strategy breakdown
    if trades:
        print("\n" + "=" * 70)
        print("STRATEGY P&L BREAKDOWN")
        print("=" * 70)

        by_strat: dict[str, list[int]] = {}
        for t in trades:
            strat = t[5] or "calibration_edge"
            if strat not in by_strat:
                by_strat[strat] = []
            by_strat[strat].append(t[4] or 0)

        for strat, pnls in sorted(by_strat.items()):
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p < 0)
            print(
                f"  {strat:<25} P&L: ${total/100:>+8.2f}  "
                f"W/L: {wins}/{losses}  "
                f"Avg: ${(total/len(pnls)/100):>+.2f}"
            )


def main():
    parser = argparse.ArgumentParser(description="KalshiEdge Backtest")
    parser.add_argument("--db", default="kalshiedge.db", help="Database path")
    parser.add_argument(
        "--extremize",
        default="1.0,1.1,1.2,1.3,1.5,1.7,2.0",
        help="Comma-separated extremization factors to test",
    )
    args = parser.parse_args()
    factors = [float(f) for f in args.extremize.split(",")]
    run_backtest(args.db, factors)


if __name__ == "__main__":
    main()
