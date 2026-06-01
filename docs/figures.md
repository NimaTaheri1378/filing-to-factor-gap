# Figures

Generated figures are written to `artifacts/figures/public/` after a smoke or full run. Each final figure is exported as PNG, PDF, and SVG.

Core public figures:

- Coverage by year: `coverage_by_year.*`
- Filing coverage heatmap: `coverage_heatmap.*`
- Gap distribution: `gap_distribution.*`
- Gap and complexity scatter: `gap_complexity_scatter.*`
- Gap decile returns: `gap_decile_bar.*`
- Long-short cumulative wealth: `long_short_returns.*`
- Rolling net return: `rolling_net_return.*`
- Event returns: `event_returns.*`
- Robustness rank IC: `robustness_rank_ic.*`

The committed figure files live outside the MkDocs source tree under `artifacts/figures/public/` so they can also be consumed directly by papers, slides, and README previews.
