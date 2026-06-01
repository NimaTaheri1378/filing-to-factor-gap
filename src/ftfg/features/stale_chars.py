from __future__ import annotations

import numpy as np
import pandas as pd


def build_stale_characteristics(compustat_panel: pd.DataFrame) -> pd.DataFrame:
    if {"stale_profitability", "stale_investment", "stale_accruals"}.issubset(compustat_panel.columns):
        return compustat_panel.copy()

    df = compustat_panel.sort_values(["permno", "date"]).copy()
    assets = df.get("at", df.get("assets", np.nan))
    lag_assets = df.groupby("permno")[assets.name].shift(1) if hasattr(assets, "name") else np.nan
    df["stale_profitability"] = df.get("ni", df.get("net_income", np.nan)) / assets
    df["stale_investment"] = (assets - lag_assets) / lag_assets
    df["stale_accruals"] = (
        df.get("ni", df.get("net_income", np.nan)) - df.get("oancf", df.get("cfo", np.nan))
    ) / assets
    if "debt" in df:
        df["stale_leverage_change"] = df.groupby("permno")["debt"].pct_change()
    return df.replace([np.inf, -np.inf], np.nan)
