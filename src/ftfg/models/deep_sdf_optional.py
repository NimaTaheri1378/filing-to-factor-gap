from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def fit_deep_model_optional(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = "ret_fwd_1m",
    epochs: int = 10,
) -> pd.DataFrame:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - optional dependency
        LOG.warning("PyTorch unavailable; returning zero deep predictions: %s", exc)
        out = test[["date", "permno", target]].copy()
        out["pred_deep"] = 0.0
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("Fitting deep model on device=%s", device)
    x_train = train[features].fillna(train[features].median()).to_numpy(dtype=np.float32)
    y_train = train[target].fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
    x_test = test[features].fillna(train[features].median()).to_numpy(dtype=np.float32)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    xb = torch.tensor(x_train, device=device)
    yb = torch.tensor(y_train, device=device)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = model(torch.tensor(x_test, device=device)).cpu().numpy().ravel()
    out = test[["date", "permno", target]].copy()
    out["pred_deep"] = preds
    return out
