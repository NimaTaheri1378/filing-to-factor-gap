from __future__ import annotations

import numpy as np
import pandas as pd


def build_complexity_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "complexity_score" in out:
        return out
    inputs = []
    for col in ["custom_tag_share", "missingness_burden", "articulation_error", "mapping_entropy"]:
        if col in out:
            inputs.append(out[col].astype(float))
    if "amendment_flag" in out:
        inputs.append(out["amendment_flag"].astype(float))
    if not inputs:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out["missingness_burden"] = out[numeric_cols].isna().mean(axis=1)
        inputs.append(out["missingness_burden"])
    score = pd.concat(inputs, axis=1).rank(pct=True).mean(axis=1)
    out["complexity_score"] = score.fillna(score.median())
    return out
