from __future__ import annotations

import numpy as np
import pandas as pd


def spearman_by_date(df: pd.DataFrame, pred: str, target: str, date_col: str = "date") -> pd.DataFrame:
    rows = []
    for date, g in df.groupby(date_col):
        valid = g[[pred, target]].dropna()
        if len(valid) < 5:
            continue
        rows.append({"date": date, "rank_ic": valid[pred].rank().corr(valid[target].rank())})
    return pd.DataFrame(rows)


def summarize_returns(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {"mean": np.nan, "vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    vol = r.std(ddof=1)
    return {
        "mean": float(r.mean() * periods_per_year),
        "vol": float(vol * np.sqrt(periods_per_year)),
        "sharpe": float((r.mean() / vol) * np.sqrt(periods_per_year)) if vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
    }


def decile_table(df: pd.DataFrame, score: str, target: str, date_col: str = "date") -> pd.DataFrame:
    work = df[[date_col, score, target]].dropna().copy()
    work["decile"] = work.groupby(date_col)[score].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    )
    return work.groupby("decile")[target].agg(["mean", "count", "std"]).reset_index()


def model_scorecard(predictions: dict[str, pd.DataFrame], target: str = "ret_fwd_1m") -> pd.DataFrame:
    rows = []
    for name, frame in predictions.items():
        pred_cols = [c for c in frame.columns if c.startswith("pred_")]
        if not pred_cols or target not in frame:
            continue
        pred = pred_cols[0]
        valid = frame[[pred, target]].dropna()
        if valid.empty:
            continue
        y = valid[target]
        x = valid[pred]
        ss_res = float(((y - x) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        rows.append(
            {
                "model": name,
                "nobs": int(len(valid)),
                "rank_ic": float(x.rank().corr(y.rank())),
                "pearson_ic": float(x.corr(y)),
                "oos_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "hit_rate": float(((x > x.median()) == (y > y.median())).mean()),
            }
        )
    return pd.DataFrame(rows)
