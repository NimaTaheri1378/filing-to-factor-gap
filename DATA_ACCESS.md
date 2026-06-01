# data access

This project is designed for two replication modes.

## Public mode

The public repository installs, tests, builds docs, and runs a synthetic smoke pipeline without WRDS:

```bash
python -m pip install -e ".[dev,ml,docs]"
python -m ftfg.cli smoke --no-deep
python -m pytest
```

## Private research mode

Private replication requires licensed access to:

| Layer | Source |
|---|---|
| Returns and implementation | WRDS CRSP monthly and daily stock files |
| Linkage | WRDS CCM |
| Standardized fundamentals | WRDS Compustat |
| As-filed fundamentals | WRDS `contrib_as_filed_financials` |
| Benchmark factors | WRDS `ff_all.fivefactors_monthly` |
| Analyst controls | WRDS IBES, when entitled |
| Filing metadata and fallback facts | SEC EDGAR submissions and companyfacts APIs |

The extraction code writes restartable Parquet shards and manifests. Reruns reuse completed WRDS and SEC caches unless a cache is missing or the command is explicitly forced.

## Credential handling

Use `~/.pgpass` with `600` permissions for WRDS on Amarel or another research host. Keep credentials, API keys, raw WRDS files, SEC JSON caches, row-level panels, logs, and private model-prediction files outside Git.
