# Results

The final private run builds a point-in-time panel from WRDS, SEC EDGAR/XBRL, CRSP, Compustat, CCM, FF factors, and IBES where available. The public repository ships only aggregate tables and figures; row-level panels and row-level model predictions remain in ignored private artifact folders.

## Sample

| Measure | Value |
|---|---:|
| Filing observations | 247,806 |
| Linked CRSP firms | 7,057 |
| Panel columns | 144 |
| First signal month | 2009-04-30 |
| Last signal month | 2026-05-31 |
| 10-Q filings | 166,923 |
| 10-K filings | 76,589 |

## Model Horse Race

| Model | OOS observations | Rank IC | Pearson IC | OOS R2 | Hit rate |
|---|---:|---:|---:|---:|---:|
| Elastic Net | 109,662 | 0.0771 | 0.0150 | -0.0027 | 0.5295 |
| LightGBM | 109,662 | 0.1130 | 0.1475 | 0.0195 | 0.5339 |
| Deep model | 109,662 | -0.0196 | -0.0024 | -5.5075 | 0.4938 |

LightGBM was fit in CPU mode after the cluster build reported that GPU tree learning was unavailable. The optional deep model ran on CUDA.

## Backtest And Alpha

| Metric | Value |
|---|---:|
| Net annualized mean | 0.6092 |
| Net annualized volatility | 0.3567 |
| Net Sharpe | 1.7078 |
| Max drawdown | -0.5629 |
| Gross FF alpha | 0.0581 |
| Gross FF alpha t-stat | 2.9483 |
| Net FF alpha | 0.0541 |
| Net FF alpha t-stat | 2.7459 |

## Robustness Slices

The robustness table reports rank IC and top-minus-bottom decile returns by sample slice. The 10-K-only slice has positive mean rank IC, while 10-Q filings are weaker in this first implementation.

Key public files:

- `artifacts/tables/public/model_scorecard.csv`
- `artifacts/tables/public/monthly_backtest_summary.csv`
- `artifacts/tables/public/factor_alpha.csv`
- `artifacts/tables/public/robustness_summary.csv`
- `artifacts/tables/public/event_study.csv`
