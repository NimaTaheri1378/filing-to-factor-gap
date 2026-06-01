from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

LOG = logging.getLogger(__name__)


def fit_lightgbm_or_fallback(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = "ret_fwd_1m",
    use_gpu: bool = True,
) -> tuple[pd.DataFrame, object]:
    try:
        import lightgbm as lgb

        base_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbose": -1,
        }
        params = dict(base_params)
        if use_gpu:
            params["device_type"] = "gpu"
            try:
                model = lgb.LGBMRegressor(**params, n_estimators=800)
                model.fit(train[features], train[target])
            except Exception as exc:
                LOG.warning("LightGBM GPU failed; retrying CPU LightGBM: %s", exc)
                model = lgb.LGBMRegressor(**base_params, n_estimators=800)
                model.fit(train[features], train[target])
        else:
            model = lgb.LGBMRegressor(**params, n_estimators=800)
            model.fit(train[features], train[target])
        fitted: object = model
        preds = model.predict(test[features])
    except Exception as exc:  # pragma: no cover - fallback depends on optional runtime
        LOG.warning("LightGBM unavailable; using sklearn fallback: %s", exc)
        fitted = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)),
            ]
        )
        fitted.fit(train[features], train[target])
        preds = fitted.predict(test[features])

    out = test[["date", "permno", target]].copy()
    out["pred_lightgbm"] = preds
    return out, fitted
