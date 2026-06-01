from __future__ import annotations

import pandas as pd


def monthly_long_short(
    panel: pd.DataFrame,
    score: str,
    ret_col: str = "ret_fwd_1m",
    date_col: str = "date",
    sector_col: str = "sic2",
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    rows = []
    prev_weights: pd.Series | None = None
    for date, g in panel.dropna(subset=[score, ret_col]).groupby(date_col):
        parts = []
        group_iter = list(g.groupby(sector_col)) if sector_col in g else [(None, g)]
        for _, sg in group_iter:
            if len(sg) < 10:
                continue
            parts.append(_long_short_weights(sg, score))
        if not parts and len(g) >= 20:
            parts.append(_long_short_weights(g, score))
        if not parts:
            continue
        weights = pd.concat(parts).groupby(level=0).sum()
        weights = weights / weights.abs().sum() * 2
        gross = float((weights * g.loc[weights.index, ret_col]).sum())
        if prev_weights is None:
            turnover = float(weights.abs().sum())
        else:
            aligned = pd.concat([prev_weights, weights], axis=1).fillna(0)
            turnover = float((aligned.iloc[:, 1] - aligned.iloc[:, 0]).abs().sum())
        net = gross - turnover * cost_bps / 10000
        rows.append({"date": date, "gross_ret": gross, "net_ret": net, "turnover": turnover})
        prev_weights = weights
    return pd.DataFrame(rows)


def _long_short_weights(frame: pd.DataFrame, score: str) -> pd.Series:
    lo = frame[score].quantile(0.1)
    hi = frame[score].quantile(0.9)
    long = frame[frame[score] >= hi]
    short = frame[frame[score] <= lo]
    w = pd.Series(0.0, index=frame.index)
    if not long.empty:
        w.loc[long.index] = 1 / len(long)
    if not short.empty:
        w.loc[short.index] = -1 / len(short)
    return w
