# Filing-to-Factor Gap

**Filing-to-Factor Gap** is a point-in-time U.S. equity asset-pricing project built around as-filed financial statements, SEC filing timestamps, standard Compustat/CRSP characteristics, and reporting-complexity frictions.

The research question is simple: do newly public as-filed characteristic shocks predict future stock returns beyond stale standardized factor inputs?

## What This Repository Ships

- deterministic WRDS schema audit and entitlement map;
- SEC EDGAR submission/XBRL metadata fetchers with public-safe rate limiting;
- point-in-time feature construction for as-filed characteristics, stale comparators, filing-to-factor gaps, complexity, and attention controls;
- event studies, Fama-MacBeth regressions, Elastic Net, LightGBM, and an optional GPU deep asset-pricing model;
- monthly long-short and filing-event backtests with costs, turnover, capacity, and FF alpha tables;
- polished public figures and tables;
- synthetic fixtures and CI so the repository is reproducible without proprietary WRDS data.

Raw WRDS data, private SEC/WRDS caches, credentials, passwords, and API keys are intentionally excluded.

## Full Run Snapshot

The private Amarel run materialized a point-in-time panel with 247,806 filing observations, 7,057 linked CRSP firms, and 144 columns spanning 10-K/10-Q filings from 2009-04-30 through 2026-05-31. Public outputs are aggregate only: row-level WRDS/CRSP-linked panels and model predictions are written under ignored private artifact folders.

Headline public artifacts:

- 16 aggregate CSV tables under `artifacts/tables/public/`, including model scorecards, FF alpha, event-study, robustness, coverage, and backtest summaries.
- 27 publication-oriented figure files under `artifacts/figures/public/` across PNG, PDF, and SVG.
- CUDA-backed deep model execution on Amarel; LightGBM retried CPU mode because the cluster LightGBM build did not include GPU tree learner support.

Selected final metrics from the private run:

| Output | Value |
|---|---:|
| LightGBM rank IC | 0.1130 |
| Elastic Net rank IC | 0.0771 |
| Net long-short annualized mean | 0.6092 |
| Net long-short annualized Sharpe | 1.7078 |
| Net FF alpha | 0.0541 |
| Net FF alpha t-stat | 2.7459 |

## Quickstart Without WRDS

```bash
python -m pip install -e ".[dev,ml]"
python -m ftfg.cli smoke --synthetic
python -m pytest
```

The smoke command creates a small synthetic fixture, runs the same feature/model/backtest machinery, and writes public-safe outputs under `artifacts/`.

## Private WRDS Run

On Amarel, run only inside the approved project workspace:

```bash
cd "/scratch/nt612/Github/Filing-to-Factor Gap"
sbatch jobs/run_schema_smoke.sbatch
sbatch jobs/run_full_pipeline.sbatch
```

WRDS credentials should come from `~/.pgpass` with `600` permissions. Do not place passwords in `.env`, config files, logs, notebooks, or Git history.

To resume after WRDS and SEC caches are complete:

```bash
bash jobs/resume_after_sec.sh
```

## Proposal Defaults

- Warm-up: 2009
- Main sample: 2010-2025
- Holdout: 2026 YTD
- Core WRDS stack: as-filed financials, Compustat, CRSP, CCM, Fama-French factors
- Optional controls: IBES and TAQ/liquidity products when entitlement is available
- Public release emphasis: visuals, tables, docs, and reproducibility rather than a manuscript
