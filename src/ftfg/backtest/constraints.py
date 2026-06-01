from __future__ import annotations

import pandas as pd


def apply_universe_filters(panel: pd.DataFrame, min_price: float = 5.0, min_dollar_volume: float = 1_000_000) -> pd.DataFrame:
    out = panel.copy()
    if "price" in out:
        out = out[out["price"].abs() >= min_price]
    if "dollar_volume" in out:
        out = out[out["dollar_volume"] >= min_dollar_volume]
    return out
