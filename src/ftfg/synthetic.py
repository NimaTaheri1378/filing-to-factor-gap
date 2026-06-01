from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_panel(n_firms: int = 240, start: str = "2018-01-31", periods: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(1378)
    dates = pd.date_range(start=start, periods=periods, freq="ME")
    permnos = np.arange(10000, 10000 + n_firms)
    rows = []
    sectors = rng.integers(10, 60, size=n_firms)
    firm_quality = rng.normal(0, 0.5, size=n_firms)
    for date in dates:
        market = rng.normal(0.005, 0.04)
        for j, permno in enumerate(permnos):
            filing_noise = rng.normal(0, 0.8)
            profitability = firm_quality[j] + rng.normal(0, 1)
            investment = rng.normal(0, 1)
            accruals = rng.normal(0, 1)
            stale_profitability = 0.65 * profitability + rng.normal(0, 0.8)
            complexity = np.clip(rng.beta(2, 5) + 0.08 * abs(filing_noise), 0, 1.5)
            gap_profitability = profitability - stale_profitability
            alpha = 0.004 * gap_profitability + 0.002 * gap_profitability * complexity - 0.002 * accruals
            ret_fwd_1m = market + alpha + rng.normal(0, 0.08)
            rows.append(
                {
                    "date": date,
                    "permno": int(permno),
                    "gvkey": f"{permno:06d}",
                    "cik": int(permno + 1_000_000),
                    "sic2": int(sectors[j]),
                    "market_equity": float(np.exp(rng.normal(20, 1.2))),
                    "dollar_volume": float(np.exp(rng.normal(15, 1.0))),
                    "profitability_af": profitability,
                    "investment_af": investment,
                    "accruals_af": accruals,
                    "leverage_change_af": rng.normal(0, 1),
                    "gross_margin_af": profitability + rng.normal(0, 0.3),
                    "stale_profitability": stale_profitability,
                    "stale_investment": 0.7 * investment + rng.normal(0, 0.8),
                    "stale_accruals": 0.7 * accruals + rng.normal(0, 0.8),
                    "complexity_score": complexity,
                    "after_close": int(rng.random() < 0.42),
                    "analyst_coverage": int(rng.poisson(6)),
                    "ret_fwd_1m": ret_fwd_1m,
                    "ret_5d": ret_fwd_1m / 4 + rng.normal(0, 0.03),
                    "ret_20d": ret_fwd_1m + rng.normal(0, 0.04),
                    "ret_60d": ret_fwd_1m * 1.8 + rng.normal(0, 0.08),
                }
            )
    return pd.DataFrame(rows)


def write_synthetic_fixture(root: Path) -> Path:
    out_dir = root / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthetic_panel.parquet"
    make_synthetic_panel().to_parquet(path, index=False)
    return path
