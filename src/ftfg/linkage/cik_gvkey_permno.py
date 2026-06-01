from __future__ import annotations

import pandas as pd


def validate_link_windows(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if {"date", "linkdt", "linkenddt"}.issubset(out.columns):
        link_end = out["linkenddt"].fillna(pd.Timestamp("2100-01-01"))
        out = out[(out["date"] >= out["linkdt"]) & (out["date"] <= link_end)]
    return out


def summarize_linkage(panel: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(panel)),
        "permnos": int(panel["permno"].nunique()) if "permno" in panel else 0,
        "gvkeys": int(panel["gvkey"].nunique()) if "gvkey" in panel else 0,
        "ciks": int(panel["cik"].nunique()) if "cik" in panel else 0,
    }
