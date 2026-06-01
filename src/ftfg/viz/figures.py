from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ftfg.evaluation.metrics import spearman_by_date


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def make_core_figures(panel: pd.DataFrame, backtest: pd.DataFrame, out_dir: Path, event: pd.DataFrame | None = None) -> dict[str, Path]:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")
    panel = panel.copy()
    for col in ["gap_composite", "complexity_score", "ret_fwd_1m", "ret_5d", "ret_20d", "ret_60d"]:
        if col in panel:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(10, 6))
    if "gap_composite" in panel:
        ax.hist(panel["gap_composite"].dropna(), bins=60, color="#2a9d8f", alpha=0.9)
    ax.set(title="Filing-to-Factor Gap Distribution", xlabel="Composite gap", ylabel="Observations")
    paths["gap_distribution"] = _save(fig, out_dir, "gap_distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    coverage = panel.assign(year=panel["date"].dt.year).groupby("year")["permno"].nunique()
    coverage.plot(kind="bar", ax=ax, color="#457b9d")
    ax.set(title="Synthetic/Private Coverage by Year", xlabel="Year", ylabel="Unique firms")
    paths["coverage_by_year"] = _save(fig, out_dir, "coverage_by_year")

    if not backtest.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        bt = backtest.sort_values("date").copy()
        bt["gross_wealth"] = (1 + bt["gross_ret"]).cumprod()
        bt["net_wealth"] = (1 + bt["net_ret"]).cumprod()
        ax.plot(bt["date"], bt["gross_wealth"], label="Gross", color="#264653")
        ax.plot(bt["date"], bt["net_wealth"], label="Net", color="#e76f51")
        ax.legend()
        ax.set(title="Long-Short Gap Strategy", xlabel="Date", ylabel="Cumulative wealth")
        paths["long_short_returns"] = _save(fig, out_dir, "long_short_returns")

        fig, ax = plt.subplots(figsize=(11, 6))
        bt["rolling_12m_net"] = bt["net_ret"].rolling(12, min_periods=6).mean() * 12
        ax.plot(bt["date"], bt["rolling_12m_net"], color="#6a994e")
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set(title="Rolling 12-Month Net Return", xlabel="Date", ylabel="Annualized return")
        paths["rolling_net_return"] = _save(fig, out_dir, "rolling_net_return")

    fig, ax = plt.subplots(figsize=(10, 6))
    if {"complexity_score", "gap_composite", "ret_fwd_1m"}.issubset(panel.columns):
        plot_df = panel.sample(min(5000, len(panel)), random_state=1378)
        points = ax.scatter(
            plot_df["gap_composite"],
            plot_df["ret_fwd_1m"],
            c=plot_df["complexity_score"],
            cmap="viridis",
            alpha=0.35,
            s=14,
            linewidths=0,
        )
        fig.colorbar(points, ax=ax, label="Complexity")
    ax.set(title="Gap, Complexity, and Forward Returns", xlabel="Composite gap", ylabel="Next-month return")
    paths["gap_complexity_scatter"] = _save(fig, out_dir, "gap_complexity_scatter")

    if "gap_composite" in panel and "ret_fwd_1m" in panel:
        dec = panel.dropna(subset=["gap_composite", "ret_fwd_1m"]).copy()
        if not dec.empty:
            dec["decile"] = dec.groupby("date")["gap_composite"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1)
            means = dec.groupby("decile")["ret_fwd_1m"].mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(means.index.astype(int), means.values, color="#457b9d")
            ax.axhline(0, color="#222222", linewidth=1)
            ax.set(title="Forward Returns by Gap Decile", xlabel="Gap decile", ylabel="Mean next-month return")
            paths["gap_decile_bar"] = _save(fig, out_dir, "gap_decile_bar")

    if event is not None and not event.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for bucket, g in event.groupby("bucket"):
            ax.plot(g["horizon_days"], g["mean_return"], marker="o", label=f"Bucket {int(bucket)}")
        ax.axhline(0, color="#222222", linewidth=1)
        ax.legend(ncol=2, fontsize=9)
        ax.set(title="Post-Filing Event Returns", xlabel="Holding horizon days", ylabel="Mean return")
        paths["event_returns"] = _save(fig, out_dir, "event_returns")

    if {"date", "form", "permno"}.issubset(panel.columns):
        cov = panel.assign(year=panel["date"].dt.year).pivot_table(index="year", columns="form", values="permno", aggfunc="count", fill_value=0)
        if not cov.empty:
            cov = cov.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
            fig, ax = plt.subplots(figsize=(10, 7))
            im = ax.imshow(cov.values, aspect="auto", cmap="YlGnBu")
            ax.set_xticks(range(len(cov.columns)), cov.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(cov.index)), cov.index.astype(str))
            ax.set(title="Filing Coverage Heatmap", xlabel="Form", ylabel="Year")
            fig.colorbar(im, ax=ax, label="Filings")
            paths["coverage_heatmap"] = _save(fig, out_dir, "coverage_heatmap")

    robust = _robustness_rank_ic(panel)
    if not robust.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(robust["spec"], robust["mean_rank_ic"], color="#5b8e7d")
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set(title="Robustness: Mean Rank IC by Sample Slice", xlabel="Mean rank IC", ylabel="")
        paths["robustness_rank_ic"] = _save(fig, out_dir, "robustness_rank_ic")

    return paths


def _robustness_rank_ic(panel: pd.DataFrame) -> pd.DataFrame:
    if not {"gap_composite", "ret_fwd_1m", "date"}.issubset(panel.columns):
        return pd.DataFrame()
    specs: list[tuple[str, pd.DataFrame]] = [("All", panel)]
    if "form" in panel:
        form = panel["form"].astype(str)
        specs.extend(
            [
                ("No amendments", panel[~form.str.endswith("/A", na=False)]),
                ("10-K", panel[form.str.startswith("10-K", na=False)]),
                ("10-Q", panel[form.str.startswith("10-Q", na=False)]),
            ]
        )
    if "market_equity" in panel:
        me = pd.to_numeric(panel["market_equity"], errors="coerce")
        specs.append(("Large", panel[me >= me.median()]))
    if "dollar_volume" in panel:
        dv = pd.to_numeric(panel["dollar_volume"], errors="coerce")
        specs.append(("Liquid", panel[dv >= dv.median()]))
    rows = []
    for spec, frame in specs:
        ic = spearman_by_date(frame, "gap_composite", "ret_fwd_1m")
        if not ic.empty:
            rows.append({"spec": spec, "mean_rank_ic": ic["rank_ic"].mean()})
    return pd.DataFrame(rows).sort_values("mean_rank_ic")
