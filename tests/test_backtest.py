from __future__ import annotations

from ftfg.backtest.monthly_sleeve import monthly_long_short
from ftfg.features import build_gap_features, build_liquidity_controls
from ftfg.synthetic import make_synthetic_panel


def test_monthly_long_short_outputs_returns() -> None:
    panel = build_liquidity_controls(build_gap_features(make_synthetic_panel(n_firms=80, periods=12)))
    bt = monthly_long_short(panel, "gap_composite")
    assert not bt.empty
    assert {"gross_ret", "net_ret", "turnover"}.issubset(bt.columns)
