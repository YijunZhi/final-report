"""Formal VAR-Granger causality method using statsmodels."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR


METHOD_NAME = "VAR-GC"
DEFAULT_PARAMS = {
    "implementation": "statsmodels.tsa.api.VAR",
    "lag_selection": "BIC",
    "max_lag": 5,
    "alpha": 0.05,
    "multiple_testing": "Benjamini-Hochberg FDR over ordered pairs",
}


def _select_lag_bic(data: pd.DataFrame, max_lag: int) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected = VAR(data).select_order(maxlags=max_lag)
    lag = selected.selected_orders.get("bic")
    if lag is None or int(lag) < 1:
        lag = 1
    return min(int(lag), max_lag)


def _best_source_lag(results, source: str, target: str, selected_lag: int) -> tuple[int, float]:
    best_lag = 1
    best_abs_t = -np.inf
    tvalues = getattr(results, "tvalues", None)
    params = getattr(results, "params", None)
    for lag in range(1, selected_lag + 1):
        row = f"L{lag}.{source}"
        score = 0.0
        if tvalues is not None and row in tvalues.index:
            score = abs(float(tvalues.loc[row, target]))
        elif params is not None and row in params.index:
            score = abs(float(params.loc[row, target]))
        if score > best_abs_t:
            best_abs_t = score
            best_lag = lag
    return best_lag, float(best_abs_t if np.isfinite(best_abs_t) else 0.0)


def infer_edges(data: pd.DataFrame, max_lag: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    frame = data.apply(pd.to_numeric, errors="coerce").dropna().astype(float)
    variables = list(frame.columns)
    selected_lag = _select_lag_bic(frame, max_lag=max_lag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = VAR(frame).fit(selected_lag)

    rows: list[dict] = []
    for source in variables:
        for target in variables:
            if source == target:
                continue
            try:
                test = results.test_causality(caused=target, causing=[source], kind="f")
                p_value = float(test.pvalue)
                stat = float(test.test_statistic)
            except Exception:
                p_value = 1.0
                stat = 0.0
            best_lag, lag_score = _best_source_lag(results, source, target, selected_lag)
            rows.append(
                {
                    "method": METHOD_NAME,
                    "source": source,
                    "target": target,
                    "lag": best_lag,
                    "selected_var_lag": selected_lag,
                    "score": stat,
                    "lag_score": lag_score,
                    "p_value": p_value,
                }
            )

    out = pd.DataFrame(rows)
    _, q_values, _, _ = multipletests(out["p_value"].to_numpy(), alpha=alpha, method="fdr_bh")
    out["q_value"] = q_values
    out["significant"] = out["q_value"] <= alpha
    return out

