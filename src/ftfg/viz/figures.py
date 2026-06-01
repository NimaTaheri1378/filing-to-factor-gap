from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from ftfg.evaluation.metrics import spearman_by_date, summarize_returns

BLUE = "#2f5d7c"
TEAL = "#2a9d8f"
GREEN = "#5b8e7d"
ORANGE = "#d96c4a"
GRAY = "#333333"
LIGHT_GRAY = "#d9dee3"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=280, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#c7ccd1",
            "grid.color": LIGHT_GRAY,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _finish_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def make_core_figures(panel: pd.DataFrame, backtest: pd.DataFrame, out_dir: Path, event: pd.DataFrame | None = None) -> dict[str, Path]:
    _set_style()
    panel = panel.copy()
    for col in ["gap_composite", "complexity_score", "ret_fwd_1m", "ret_5d", "ret_20d", "ret_60d"]:
        if col in panel:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    paths: dict[str, Path] = {}

    if not backtest.empty:
        fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        bt = backtest.sort_values("date").copy()
        bt["net_wealth"] = (1 + bt["net_ret"]).cumprod()
        stats = summarize_returns(bt["net_ret"])
        ax.plot(bt["date"], bt["net_wealth"], color=BLUE, linewidth=2.5)
        _finish_axis(ax, "Headline Result: Net Long-Short Wealth", "Date", "Growth of $1")
        ax.text(
            0.01,
            0.96,
            f"Annualized return {stats['mean']:.1%} | Sharpe {stats['sharpe']:.2f} | Max drawdown {stats['max_drawdown']:.1%}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color=GRAY,
        )
        paths["headline_performance"] = _save(fig, out_dir, "headline_performance")

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    if "gap_composite" in panel:
        gap = panel["gap_composite"].dropna()
        lo, hi = gap.quantile([0.005, 0.995])
        ax.hist(gap.clip(lo, hi), bins=60, color=TEAL, alpha=0.92)
    _finish_axis(ax, "Distribution of Filing-to-Factor Gaps", "Composite gap", "Observations")
    paths["gap_distribution"] = _save(fig, out_dir, "gap_distribution")

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    coverage = panel.assign(year=panel["date"].dt.year).groupby("year")["permno"].nunique()
    coverage.plot(kind="bar", ax=ax, color=BLUE, width=0.82)
    _finish_axis(ax, "Filing Coverage by Year", "Year", "Unique linked firms")
    ax.tick_params(axis="x", rotation=45)
    paths["coverage_by_year"] = _save(fig, out_dir, "coverage_by_year")

    if not backtest.empty:
        fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        bt = backtest.sort_values("date").copy()
        bt["gross_wealth"] = (1 + bt["gross_ret"]).cumprod()
        bt["net_wealth"] = (1 + bt["net_ret"]).cumprod()
        ax.plot(bt["date"], bt["gross_wealth"], label="Gross", color=BLUE, linewidth=2.0)
        ax.plot(bt["date"], bt["net_wealth"], label="Net", color=ORANGE, linewidth=2.0)
        ax.legend(frameon=False, loc="upper left")
        _finish_axis(ax, "Long-Short Gap Portfolio", "Date", "Growth of $1")
        paths["long_short_returns"] = _save(fig, out_dir, "long_short_returns")

        fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
        bt["rolling_12m_net"] = bt["net_ret"].rolling(12, min_periods=6).mean() * 12
        ax.plot(bt["date"], bt["rolling_12m_net"], color=GREEN, linewidth=2.0)
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        _finish_axis(ax, "Rolling 12-Month Net Return", "Date", "Annualized return")
        paths["rolling_net_return"] = _save(fig, out_dir, "rolling_net_return")

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    if {"complexity_score", "gap_composite", "ret_fwd_1m"}.issubset(panel.columns):
        plot_df = panel.dropna(subset=["gap_composite", "ret_fwd_1m", "complexity_score"]).sample(min(6000, len(panel)), random_state=1378)
        ylo, yhi = plot_df["ret_fwd_1m"].quantile([0.01, 0.99])
        plot_df = plot_df[plot_df["ret_fwd_1m"].between(ylo, yhi)]
        points = ax.scatter(
            plot_df["gap_composite"],
            plot_df["ret_fwd_1m"],
            c=plot_df["complexity_score"],
            cmap="viridis",
            alpha=0.35,
            s=14,
            linewidths=0,
        )
        fig.colorbar(points, ax=ax, label="Complexity score")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _finish_axis(ax, "Gap, Complexity, and Forward Returns", "Composite gap", "Next-month return")
    paths["gap_complexity_scatter"] = _save(fig, out_dir, "gap_complexity_scatter")

    if "gap_composite" in panel and "ret_fwd_1m" in panel:
        dec = panel.dropna(subset=["gap_composite", "ret_fwd_1m"]).copy()
        if not dec.empty:
            dec["decile"] = dec.groupby("date")["gap_composite"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1)
            means = dec.groupby("decile")["ret_fwd_1m"].mean()
            fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
            colors = [ORANGE if value < 0 else BLUE for value in means.values]
            ax.bar(means.index.astype(int), means.values, color=colors, width=0.78)
            ax.axhline(0, color=GRAY, linewidth=1)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            _finish_axis(ax, "Forward Returns by Gap Decile", "Gap decile", "Mean next-month return")
            paths["gap_decile_bar"] = _save(fig, out_dir, "gap_decile_bar")

    if event is not None and not event.empty:
        fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
        for bucket, g in event.groupby("bucket"):
            ax.plot(g["horizon_days"], g["mean_return"], marker="o", label=f"Bucket {int(bucket)}")
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.legend(ncol=3, fontsize=8, frameon=False)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        _finish_axis(ax, "Post-Filing Event Returns", "Holding horizon days", "Mean return")
        paths["event_returns"] = _save(fig, out_dir, "event_returns")

    if {"date", "form", "permno"}.issubset(panel.columns):
        cov = panel.assign(year=panel["date"].dt.year).pivot_table(index="year", columns="form", values="permno", aggfunc="count", fill_value=0)
        if not cov.empty:
            cov = cov.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
            fig, ax = plt.subplots(figsize=(9, 6.2), constrained_layout=True)
            im = ax.imshow(cov.values, aspect="auto", cmap="YlGnBu")
            ax.set_xticks(range(len(cov.columns)), cov.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(cov.index)), cov.index.astype(str))
            _finish_axis(ax, "Filing Coverage Heatmap", "Form", "Year")
            fig.colorbar(im, ax=ax, label="Filings")
            paths["coverage_heatmap"] = _save(fig, out_dir, "coverage_heatmap")

    robust = _robustness_rank_ic(panel)
    if not robust.empty:
        fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
        colors = [ORANGE if value < 0 else GREEN for value in robust["mean_rank_ic"]]
        ax.barh(robust["spec"], robust["mean_rank_ic"], color=colors)
        ax.axvline(0, color=GRAY, linewidth=1)
        _finish_axis(ax, "Robustness: Mean Rank IC", "Mean rank IC", "")
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
