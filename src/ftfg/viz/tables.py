from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ftfg.evaluation.metrics import (
    decile_table,
    model_scorecard,
    spearman_by_date,
    summarize_returns,
)


def write_core_tables(
    panel: pd.DataFrame,
    backtest: pd.DataFrame,
    out_dir: Path,
    model_outputs: dict[str, pd.DataFrame] | None = None,
    event: pd.DataFrame | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    coverage = (
        panel.assign(year=panel["date"].dt.year)
        .groupby("year")
        .agg(
            n_obs=("permno", "size"),
            n_firms=("permno", "nunique"),
            n_filings=("adsh", "nunique") if "adsh" in panel else ("permno", "size"),
            avg_complexity=("complexity_score", "mean"),
            avg_gap=("gap_composite", "mean") if "gap_composite" in panel else ("permno", "size"),
        )
        .reset_index()
    )
    paths["coverage"] = out_dir / "coverage_by_year.csv"
    coverage.to_csv(paths["coverage"], index=False)

    if "gap_composite" in panel:
        deciles = decile_table(panel, "gap_composite", "ret_fwd_1m")
        paths["gap_deciles"] = out_dir / "gap_decile_returns.csv"
        deciles.to_csv(paths["gap_deciles"], index=False)
        ic = spearman_by_date(panel, "gap_composite", "ret_fwd_1m")
        paths["rank_ic"] = out_dir / "rank_ic.csv"
        ic.to_csv(paths["rank_ic"], index=False)
        cols = [c for c in ["gap_composite", "complexity_score", "gap_x_complexity", "ret_fwd_1m", "ret_5d", "ret_20d", "ret_60d"] if c in panel]
        paths["feature_summary"] = out_dir / "feature_summary.csv"
        panel[cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.rename_axis("feature").reset_index().to_csv(paths["feature_summary"], index=False)
        paths["missingness"] = out_dir / "feature_missingness.csv"
        panel[cols].isna().mean().rename_axis("feature").reset_index(name="missing_rate").to_csv(paths["missingness"], index=False)
        robustness = _robustness_summary(panel)
        if not robustness.empty:
            paths["robustness"] = out_dir / "robustness_summary.csv"
            robustness.to_csv(paths["robustness"], index=False)
        if "complexity_score" in panel:
            work = panel.dropna(subset=["gap_composite", "complexity_score", "ret_fwd_1m"]).copy()
            if not work.empty:
                work["gap_quintile"] = work.groupby("date")["gap_composite"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") + 1)
                work["complexity_quintile"] = work.groupby("date")["complexity_score"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") + 1)
                paths["gap_complexity_sorts"] = out_dir / "gap_complexity_sorts.csv"
                work.groupby(["gap_quintile", "complexity_quintile"])["ret_fwd_1m"].agg(["mean", "count", "std"]).reset_index().to_csv(paths["gap_complexity_sorts"], index=False)

    if not backtest.empty:
        summary = pd.DataFrame([summarize_returns(backtest["net_ret"])])
        paths["backtest_summary"] = out_dir / "monthly_backtest_summary.csv"
        summary.to_csv(paths["backtest_summary"], index=False)
        paths["monthly_backtest"] = out_dir / "monthly_backtest.csv"
        backtest.to_csv(paths["monthly_backtest"], index=False)
        rolling = backtest.sort_values("date").copy()
        rolling["rolling_12m_net"] = rolling["net_ret"].rolling(12, min_periods=6).mean() * 12
        rolling["rolling_12m_vol"] = rolling["net_ret"].rolling(12, min_periods=6).std() * (12 ** 0.5)
        rolling["rolling_12m_sharpe"] = rolling["rolling_12m_net"] / rolling["rolling_12m_vol"]
        paths["rolling_backtest"] = out_dir / "rolling_backtest.csv"
        rolling.to_csv(paths["rolling_backtest"], index=False)
    if event is not None and not event.empty:
        paths["event_study"] = out_dir / "event_study.csv"
        event.to_csv(paths["event_study"], index=False)
    if model_outputs:
        scorecard = model_scorecard(model_outputs)
        paths["model_scorecard"] = out_dir / "model_scorecard.csv"
        scorecard.to_csv(paths["model_scorecard"], index=False)
    paths["dataset_table"] = out_dir / "dataset_table.csv"
    pd.DataFrame(
        [
            {"layer": "as_filed_fundamentals", "source": "WRDS contrib_as_filed_financials + SEC companyfacts fallback", "public": False},
            {"layer": "filing_metadata", "source": "SEC submissions API", "public": True},
            {"layer": "standardized_fundamentals", "source": "WRDS Compustat", "public": False},
            {"layer": "returns_and_linking", "source": "WRDS CRSP + CCM", "public": False},
            {"layer": "factors", "source": "WRDS ff_all / French factors", "public": True},
            {"layer": "attention_controls", "source": "WRDS IBES when entitled", "public": False},
        ]
    ).to_csv(paths["dataset_table"], index=False)
    return paths


def _robustness_summary(panel: pd.DataFrame) -> pd.DataFrame:
    specs: list[tuple[str, pd.DataFrame]] = [("all_filings", panel)]
    if "form" in panel:
        form = panel["form"].astype(str)
        specs.extend(
            [
                ("exclude_amendments", panel[~form.str.endswith("/A", na=False)]),
                ("10k_only", panel[form.str.startswith("10-K", na=False)]),
                ("10q_only", panel[form.str.startswith("10-Q", na=False)]),
            ]
        )
    if "market_equity" in panel:
        me = pd.to_numeric(panel["market_equity"], errors="coerce")
        specs.append(("large_firm_above_median_me", panel[me >= me.median()]))
    if "dollar_volume" in panel:
        dv = pd.to_numeric(panel["dollar_volume"], errors="coerce")
        specs.append(("liquid_above_median_dollar_volume", panel[dv >= dv.median()]))

    rows = []
    for spec, frame in specs:
        valid = frame.dropna(subset=["gap_composite", "ret_fwd_1m"]).copy()
        if valid.empty:
            continue
        ic = spearman_by_date(valid, "gap_composite", "ret_fwd_1m")
        dec = decile_table(valid, "gap_composite", "ret_fwd_1m")
        spread = np.nan
        if {1, 10}.issubset(set(dec["decile"].dropna().astype(int))):
            high = float(dec.loc[dec["decile"].astype(int) == 10, "mean"].iloc[0])
            low = float(dec.loc[dec["decile"].astype(int) == 1, "mean"].iloc[0])
            spread = high - low
        rows.append(
            {
                "spec": spec,
                "n_obs": int(len(valid)),
                "n_firms": int(valid["permno"].nunique()) if "permno" in valid else np.nan,
                "n_months": int(len(ic)),
                "mean_rank_ic": float(ic["rank_ic"].mean()) if not ic.empty else np.nan,
                "rank_ic_t": float(ic["rank_ic"].mean() / (ic["rank_ic"].std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 1 and ic["rank_ic"].std(ddof=1) > 0 else np.nan,
                "top_minus_bottom_decile_return": spread,
            }
        )
    return pd.DataFrame(rows)
