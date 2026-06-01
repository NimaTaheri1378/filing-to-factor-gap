from __future__ import annotations

import pandas as pd


def simple_attribution(panel: pd.DataFrame, score: str, target: str = "ret_fwd_1m") -> pd.DataFrame:
    cols = [c for c in ["complexity_score", "after_close", "analyst_coverage", "log_market_equity"] if c in panel]
    rows = []
    for col in cols:
        rows.append({"feature": col, "corr_with_score": panel[[score, col]].corr().iloc[0, 1], "corr_with_return": panel[[target, col]].corr().iloc[0, 1]})
    return pd.DataFrame(rows)
