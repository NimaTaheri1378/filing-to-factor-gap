# Data Access

Private replication requires WRDS access to CRSP, Compustat, CCM, as-filed financials, and benchmark factors. Optional IBES and TAQ/liquidity products improve controls and implementation diagnostics.

SEC EDGAR/XBRL endpoints are public and require a descriptive User-Agent.

## Data Sources Used

| Layer | Source | Publicly redistributed? |
|---|---|---|
| As-filed fundamentals | WRDS `contrib_as_filed_financials` plus SEC companyfacts fallback | No |
| Filing metadata | SEC submissions API | Derived aggregates only |
| Returns and implementation | WRDS CRSP monthly and daily stock files | No |
| Linkage | WRDS CCM | No |
| Standardized fundamentals | WRDS Compustat annual fundamentals | No |
| Benchmark factors | WRDS `ff_all.fivefactors_monthly` | Aggregate alpha output only |
| Analyst controls | WRDS IBES EPS summary | No |

The extraction layer writes restartable Parquet shards and manifests. Completed shards are reused on rerun so WRDS and SEC are not queried again unless a cache is missing or explicitly forced.

## Credential Discipline

Use `~/.pgpass` on Amarel with `600` permissions for WRDS. API keys, passwords, private caches, raw JSON, and private Parquet files must not be committed.
