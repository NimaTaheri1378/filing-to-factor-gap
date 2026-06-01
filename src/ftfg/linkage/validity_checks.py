from __future__ import annotations

import pandas as pd


def assert_no_future_features(panel: pd.DataFrame, feature_time_col: str = "feature_timestamp", activation_col: str = "activation_timestamp") -> None:
    if {feature_time_col, activation_col}.issubset(panel.columns):
        bad = panel[pd.to_datetime(panel[feature_time_col]) > pd.to_datetime(panel[activation_col])]
        if not bad.empty:
            raise AssertionError(f"{len(bad)} features occur after activation timestamp")


def duplicate_key_report(panel: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return panel.groupby(keys).size().reset_index(name="n").query("n > 1")
