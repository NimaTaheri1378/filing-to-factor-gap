from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fama_macbeth(
    panel: pd.DataFrame,
    y: str,
    xvars: list[str],
    date_col: str = "date",
    min_obs: int = 50,
) -> pd.DataFrame:
    rows = []
    for date, g in panel.groupby(date_col):
        cols = [y, *xvars]
        valid = g[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < max(min_obs, len(xvars) + 5):
            continue
        res = sm.OLS(valid[y].astype(float), sm.add_constant(valid[xvars].astype(float))).fit()
        row = {"date": date, "nobs": int(res.nobs)}
        row.update({k: float(v) for k, v in res.params.items()})
        rows.append(row)
    coefs = pd.DataFrame(rows)
    if coefs.empty:
        return coefs
    summary_rows = []
    for col in ["const", *xvars]:
        if col not in coefs:
            continue
        s = coefs[col].dropna()
        se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
        summary_rows.append(
            {
                "term": col,
                "mean_coef": float(s.mean()),
                "t_stat": float(s.mean() / se) if se and se > 0 else np.nan,
                "n_months": int(len(s)),
            }
        )
    return pd.DataFrame(summary_rows)
