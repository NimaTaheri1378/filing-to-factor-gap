from __future__ import annotations

import pandas as pd


def event_sleeve(panel: pd.DataFrame, score: str = "gap_composite") -> pd.DataFrame:
    rows = []
    for horizon, ret_col in [(5, "ret_5d"), (20, "ret_20d"), (60, "ret_60d")]:
        if ret_col not in panel:
            continue
        work = panel.dropna(subset=[score, ret_col]).copy()
        if work.empty:
            continue
        work["bucket"] = work.groupby("date")[score].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") + 1
        )
        summary = work.groupby("bucket")[ret_col].agg(mean_return="mean", nobs="size", volatility="std").reset_index()
        summary["horizon_days"] = horizon
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
