from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_mean(series: pd.Series, block: int = 6, n_boot: int = 1000, seed: int = 1378) -> dict[str, float]:
    x = series.dropna().to_numpy()
    if len(x) == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan}
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(x) - block + 1))
    draws = []
    for _ in range(n_boot):
        sample = []
        while len(sample) < len(x):
            st = int(rng.choice(starts))
            sample.extend(x[st : st + block])
        draws.append(np.mean(sample[: len(x)]))
    return {
        "mean": float(np.mean(x)),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
    }
