from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    if float(std) == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def build_gap_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    pairs = {
        "profitability": ("profitability_af", "stale_profitability"),
        "investment": ("investment_af", "stale_investment"),
        "accruals": ("accruals_af", "stale_accruals"),
    }
    for name, (as_filed, stale) in pairs.items():
        if as_filed in out and stale in out:
            raw = f"gap_{name}"
            z = f"gap_{name}_z"
            out[raw] = out[as_filed] - out[stale]
            group_cols = ["date"]
            if "sic2" in out:
                group_cols.append("sic2")
            out[z] = out.groupby(group_cols, group_keys=False)[raw].transform(_zscore)
    gap_cols = [c for c in out.columns if c.startswith("gap_") and c.endswith("_z")]
    if gap_cols:
        out["gap_composite"] = out[gap_cols].mean(axis=1)
        if "complexity_score" in out:
            out["gap_x_complexity"] = out["gap_composite"] * out["complexity_score"]
    return out
