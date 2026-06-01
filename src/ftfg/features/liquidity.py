from __future__ import annotations

import numpy as np
import pandas as pd


def build_liquidity_controls(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "dollar_volume" in out:
        out["log_dollar_volume"] = np.log1p(out["dollar_volume"].clip(lower=0))
    if "market_equity" in out:
        out["log_market_equity"] = np.log1p(out["market_equity"].clip(lower=0))
    if "bid_ask_spread" not in out:
        vol = out.get("dollar_volume", pd.Series(1.0, index=out.index)).clip(lower=1)
        out["bid_ask_spread_proxy"] = 0.01 / np.sqrt(np.log1p(vol))
    return out
