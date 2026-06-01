# Methodology

The pipeline activates signals at SEC acceptance time, builds as-filed characteristics from filing-level facts, compares them to stale standardized Compustat characteristics, and tests the resulting gap with event studies, cross-sectional regressions, machine learning, and portfolio backtests.

## Timing

The unit of observation is a public filing. As-filed fundamentals are activated at filing availability, then mapped to the next monthly signal date for cross-sectional prediction. Stale Compustat characteristics are merged strictly backward relative to filing date, and CCM links are filtered by validity windows before joining to CRSP PERMNOs.

## Feature Blocks

- As-filed fundamentals: profitability, cash-flow quality, investment, accruals, leverage, cash, R&D, SG&A, and financing measures.
- Filing-to-factor gaps: month/industry-standardized differences between as-filed characteristics and the latest stale standardized Compustat counterpart.
- Complexity: SEC/WRDS structure fields when available, with SEC companyfacts fallback measures for missingness, mapping burden, amendments, and articulation.
- Attention and implementation: size, dollar volume, bid-ask proxy, analyst coverage, after-close filing flag, and event-window returns.

## Estimation Ladder

The package runs Fama-MacBeth regressions first, then Elastic Net and LightGBM, and finally an optional PyTorch deep model. The portfolio engine builds a monthly sector-neutral long-short sleeve and a filing-event sleeve at 5, 20, and 60 trading-day horizons.

## Public-Safe Boundary

Aggregate tables and figures are public. Raw WRDS extracts, SEC JSON caches, linked row-level panels, and row-level model predictions are private artifacts and are excluded from Git.
