from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ftfg.backtest.event_sleeve import event_sleeve
from ftfg.backtest.monthly_sleeve import monthly_long_short
from ftfg.config import load_config, project_paths
from ftfg.evaluation.ff_alpha import factor_alpha
from ftfg.features import build_complexity_features, build_gap_features, build_liquidity_controls
from ftfg.features.real_panel import build_real_analysis_panel
from ftfg.manifests import write_manifest
from ftfg.models.deep_sdf_optional import fit_deep_model_optional
from ftfg.models.elastic_net import fit_elastic_net
from ftfg.models.fm import fama_macbeth
from ftfg.models.lightgbm_ranker import fit_lightgbm_or_fallback
from ftfg.synthetic import write_synthetic_fixture
from ftfg.viz.figures import make_core_figures
from ftfg.viz.tables import write_core_tables

LOG = logging.getLogger(__name__)


FEATURES = [
    "gap_profitability_z",
    "gap_investment_z",
    "gap_accruals_z",
    "gap_composite",
    "gap_x_complexity",
    "complexity_score",
    "after_close",
    "analyst_coverage",
    "log_market_equity",
    "log_dollar_volume",
    "bid_ask_spread_proxy",
]


NUMERIC_PANEL_COLUMNS = [
    *FEATURES,
    "ret_fwd_1m",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "sic2",
]


def coerce_panel_numeric(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for col in NUMERIC_PANEL_COLUMNS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def load_or_make_panel(config_path: str | Path, synthetic: bool = True) -> pd.DataFrame:
    cfg = load_config(config_path)
    paths = project_paths(cfg)
    paths.ensure()
    if synthetic:
        panel_path = write_synthetic_fixture(paths.root)
    else:
        panel_path = paths.data_root / "processed" / "analysis_panel.parquet"
        if not panel_path.exists():
            panel_path = build_real_analysis_panel(paths.data_root, paths.manifest_dir)
    if not panel_path.exists():
        raise FileNotFoundError(f"Analysis panel not found: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    return coerce_panel_numeric(panel)


def build_analysis_panel(config_path: str | Path, synthetic: bool = True) -> pd.DataFrame:
    cfg = load_config(config_path)
    paths = project_paths(cfg)
    paths.ensure()
    panel = load_or_make_panel(config_path, synthetic=synthetic)
    if synthetic:
        panel = build_complexity_features(panel)
        panel = build_gap_features(panel)
        panel = build_liquidity_controls(panel)
    out = paths.data_root / "processed" / "analysis_panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if synthetic:
        panel.to_parquet(out, index=False)
    write_manifest(paths.manifest_dir / "analysis_panel_manifest.json", {"kind": "analysis_panel", "rows": int(len(panel)), "columns": list(panel.columns)})
    LOG.info("Wrote analysis panel: %s", out)
    return panel


def run_model_stack(panel: pd.DataFrame, config_path: str | Path, include_deep: bool = True) -> dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    train_end = pd.Timestamp(cfg["sample"]["train_end"])
    valid_end = pd.Timestamp(cfg["sample"]["validation_end"])
    features = [c for c in FEATURES if c in panel.columns]
    usable = coerce_panel_numeric(panel)
    for col in ["ret_fwd_1m", *features]:
        usable[col] = pd.to_numeric(usable[col], errors="coerce").astype("float64")
    usable = usable.dropna(subset=["ret_fwd_1m"])
    train = usable[usable["date"] <= valid_end]
    test = usable[usable["date"] > train_end]
    if train.empty or test.empty:
        ordered_dates = sorted(usable["date"].dropna().unique())
        if len(ordered_dates) >= 2:
            split = ordered_dates[max(1, int(len(ordered_dates) * 0.6)) - 1]
            train = usable[usable["date"] <= split]
            test = usable[usable["date"] > split]
        else:
            train = usable.sample(frac=0.7, random_state=1378) if len(usable) else usable
            test = usable.drop(train.index)
    if len(train) < 20 or len(test) < 5 or not features:
        fallback = usable[["date", "permno", "ret_fwd_1m"]].copy()
        fallback["pred_lightgbm"] = fallback["ret_fwd_1m"].mean() if not fallback.empty else 0.0
        elastic = fallback.rename(columns={"pred_lightgbm": "pred_elastic_net"})
        return {"fm": pd.DataFrame(), "elastic_net": elastic, "lightgbm": fallback}
    outputs: dict[str, pd.DataFrame] = {}
    outputs["fm"] = fama_macbeth(panel, "ret_fwd_1m", [c for c in ["gap_composite", "complexity_score", "gap_x_complexity", "log_market_equity"] if c in panel])
    outputs["elastic_net"], _ = fit_elastic_net(train, test, features)
    outputs["lightgbm"], _ = fit_lightgbm_or_fallback(train, test, features, use_gpu=True)
    if include_deep:
        outputs["deep"] = fit_deep_model_optional(train, test, features, epochs=5)
    return outputs


def run_outputs(config_path: str | Path, synthetic: bool = True, include_deep: bool = True) -> dict[str, Path]:
    cfg = load_config(config_path)
    paths = project_paths(cfg)
    paths.ensure()
    panel = build_analysis_panel(config_path, synthetic=synthetic)
    model_outputs = run_model_stack(panel, config_path, include_deep=include_deep)
    private_model_dir = paths.artifact_root / "private" / "model_predictions"
    private_model_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in model_outputs.items():
        out = paths.public_table_dir / f"model_{name}.csv" if name == "fm" else private_model_dir / f"model_{name}.csv"
        frame.to_csv(out, index=False)
    for stale_name in ["model_elastic_net.csv", "model_lightgbm.csv", "model_deep.csv"]:
        stale_path = paths.public_table_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    score = "gap_composite"
    if "pred_lightgbm" in model_outputs.get("lightgbm", pd.DataFrame()).columns:
        preds = model_outputs["lightgbm"][["date", "permno", "pred_lightgbm"]]
        panel = panel.merge(preds, on=["date", "permno"], how="left")
        score = "pred_lightgbm"
    backtest = monthly_long_short(panel.dropna(subset=[score]), score=score)
    event = event_sleeve(panel, score="gap_composite")
    event.to_csv(paths.public_table_dir / "event_sleeve.csv", index=False)
    tables = write_core_tables(panel, backtest, paths.public_table_dir, model_outputs=model_outputs, event=event)
    factor_alpha_path = _write_factor_alpha(backtest, paths)
    if factor_alpha_path is not None:
        tables["factor_alpha"] = factor_alpha_path
    figures = make_core_figures(panel, backtest, paths.public_figure_dir, event=event)
    write_manifest(
        paths.manifest_dir / "run_outputs_manifest.json",
        {
            "kind": "run_outputs",
            "synthetic": synthetic,
            "tables": {k: _repo_relative(v, paths.root) for k, v in tables.items()},
            "figures": {k: _repo_relative(v, paths.root) for k, v in figures.items()},
            "models": list(model_outputs),
        },
    )
    return {**tables, **figures}


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_factor_alpha(backtest: pd.DataFrame, paths) -> Path | None:
    factor_path = paths.data_root / "raw" / "wrds" / "ff" / "fivefactors_monthly.parquet"
    if backtest.empty or not factor_path.exists():
        return None
    factors = pd.read_parquet(factor_path)
    factors.columns = [str(c).lower() for c in factors.columns]
    if "date" not in factors:
        return None
    factors["date"] = pd.to_datetime(factors["date"]).dt.to_period("M").dt.to_timestamp("M")
    returns = backtest[["date", "gross_ret", "net_ret"]].copy()
    returns["date"] = pd.to_datetime(returns["date"]).dt.to_period("M").dt.to_timestamp("M")
    rows = []
    for ret_col in ["gross_ret", "net_ret"]:
        stats = factor_alpha(returns[["date", ret_col]].rename(columns={ret_col: "ret"}), factors)
        stats["portfolio_return"] = ret_col
        rows.append(stats)
    out = paths.public_table_dir / "factor_alpha.csv"
    pd.DataFrame(rows)[["portfolio_return", "alpha", "t_alpha", "nobs"]].to_csv(out, index=False)
    return out
