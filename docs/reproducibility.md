# Reproducibility

The public repository includes synthetic fixtures and tests. Private runs cache WRDS and SEC extracts as Parquet with manifests, query metadata, row counts, and validation diagnostics.

## Public Smoke Mode

```bash
python -m pip install -e ".[dev,ml]"
python -m ftfg.cli smoke --no-deep
python -m pytest
```

Smoke mode uses synthetic data and exercises the same feature, model, backtest, table, and figure code paths without requiring WRDS.

## Private Full Mode

```bash
python -m ftfg.cli schema-audit
python -m ftfg.cli pull-wrds
python -m ftfg.cli pull-sec
python -m ftfg.cli build-real-panel
python -m ftfg.cli full --include-deep
```

On Amarel, the SLURM scripts under `jobs/` run the same commands inside the project workspace. `jobs/resume_after_sec.sh` resumes from completed WRDS and SEC caches and runs only panel construction plus final outputs.

## Final Run Audit

The final run wrote:

- `artifacts/manifests/real_analysis_panel_manifest.json`
- `artifacts/manifests/analysis_panel_manifest.json`
- `artifacts/manifests/run_outputs_manifest.json`
- public aggregate tables in `artifacts/tables/public/`
- public figures in `artifacts/figures/public/`

Private raw data and row-level model predictions remain excluded by `.gitignore`.
