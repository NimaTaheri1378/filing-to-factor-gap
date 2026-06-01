from __future__ import annotations

from ftfg.features import build_complexity_features, build_gap_features, build_liquidity_controls
from ftfg.synthetic import make_synthetic_panel


def test_gap_features_have_composite() -> None:
    panel = make_synthetic_panel(n_firms=20, periods=4)
    out = build_gap_features(panel)
    assert "gap_composite" in out
    assert out["gap_composite"].notna().all()


def test_complexity_and_liquidity_controls() -> None:
    panel = make_synthetic_panel(n_firms=20, periods=4).drop(columns=["complexity_score"])
    out = build_liquidity_controls(build_complexity_features(panel))
    assert "complexity_score" in out
    assert "log_dollar_volume" in out
