"""Generate calibration report from the portfolio database."""

import asyncio

from kalshiedge.calibration import generate_report
from kalshiedge.portfolio import PortfolioStore


async def main() -> None:
    store = PortfolioStore()
    await store.initialize()
    report = await generate_report(store)
    print(report)
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
