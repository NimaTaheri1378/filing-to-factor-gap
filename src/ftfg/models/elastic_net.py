from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_elastic_net(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = "ret_fwd_1m",
) -> tuple[pd.DataFrame, Pipeline]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, -1, 8), cv=5)),
        ]
    )
    pipe.fit(train[features], train[target])
    pred = test[["date", "permno", target]].copy()
    pred["pred_elastic_net"] = pipe.predict(test[features])
    return pred, pipe
