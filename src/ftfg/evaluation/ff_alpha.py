from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def factor_alpha(returns: pd.DataFrame, factors: pd.DataFrame, ret_col: str = "ret") -> dict[str, float]:
    merged = returns.merge(factors, on="date", how="inner")
    if merged.empty:
        return {"alpha": np.nan, "t_alpha": np.nan, "nobs": 0}
    factor_cols = [c for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd"] if c in merged.columns]
    for col in [ret_col, *factor_cols, "rf"]:
        if col in merged:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype(float)
    merged = merged.dropna(subset=[ret_col, *factor_cols])
    if merged.empty:
        return {"alpha": np.nan, "t_alpha": np.nan, "nobs": 0}
    for col in [*factor_cols, "rf"]:
        if col in merged and merged[col].abs().median() > 0.5:
            merged[col] = merged[col] / 100
    y = merged[ret_col] - merged["rf"] if "rf" in merged else merged[ret_col]
    x = sm.add_constant(merged[factor_cols]) if factor_cols else np.ones((len(merged), 1))
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return {
        "alpha": float(model.params.iloc[0]),
        "t_alpha": float(model.tvalues.iloc[0]),
        "nobs": int(model.nobs),
    }
