from __future__ import annotations

import pandas as pd
import pytest

from ftfg.linkage.validity_checks import assert_no_future_features, duplicate_key_report


def test_no_future_features_raises() -> None:
    panel = pd.DataFrame(
        {
            "feature_timestamp": [pd.Timestamp("2020-01-02")],
            "activation_timestamp": [pd.Timestamp("2020-01-01")],
        }
    )
    with pytest.raises(AssertionError):
        assert_no_future_features(panel)


def test_duplicate_report() -> None:
    panel = pd.DataFrame({"permno": [1, 1, 2], "date": [1, 1, 1]})
    dupes = duplicate_key_report(panel, ["permno", "date"])
    assert len(dupes) == 1
