from __future__ import annotations

import numpy as np
import pandas as pd


def map_as_filed_characteristics(facts: pd.DataFrame) -> pd.DataFrame:
    """Map filing facts into a compact characteristic panel.

    This accepts either a normalized WRDS/SEC fact table or the synthetic panel.
    Real extraction code should produce columns such as assets, sales, cogs,
    operating_income, net_income, cfo, capex, debt, equity, and shares.
    """
    if {"profitability_af", "investment_af", "accruals_af"}.issubset(facts.columns):
        return facts.copy()

    df = facts.copy()
    assets = df.get("assets", np.nan)
    lag_assets = df.groupby("permno")["assets"].shift(1) if "assets" in df else np.nan
    sales = df.get("sales", np.nan)
    cogs = df.get("cogs", np.nan)
    net_income = df.get("net_income", np.nan)
    cfo = df.get("cfo", np.nan)

    df["profitability_af"] = net_income / assets
    df["gross_margin_af"] = (sales - cogs) / assets
    df["investment_af"] = (assets - lag_assets) / lag_assets
    df["accruals_af"] = (net_income - cfo) / assets
    if "debt" in df:
        df["leverage_change_af"] = df.groupby("permno")["debt"].pct_change()
    return df.replace([np.inf, -np.inf], np.nan)
