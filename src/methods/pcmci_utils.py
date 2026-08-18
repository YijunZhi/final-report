"""Shared helpers for formal tigramite PCMCI methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI


def run_pcmci_edges(
    data: pd.DataFrame,
    cond_ind_test,
    method_name: str,
    max_lag: int = 5,
    alpha: float = 0.05,
    pc_alpha: float | None = None,
) -> pd.DataFrame:
    frame = data.apply(pd.to_numeric, errors="coerce").dropna().astype(float)
    variables = list(frame.columns)
    dataframe = pp.DataFrame(frame.to_numpy(), var_names=variables)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    results = pcmci.run_pcmci(tau_min=1, tau_max=max_lag, pc_alpha=alpha if pc_alpha is None else pc_alpha)

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    try:
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=p_matrix,
            tau_min=1,
            tau_max=max_lag,
            fdr_method="fdr_bh",
        )
    except Exception:
        q_matrix = np.array(p_matrix, copy=True)
        flat_indices = []
        flat_p = []
        n_vars = len(variables)
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                for tau in range(1, max_lag + 1):
                    flat_indices.append((i, j, tau))
                    flat_p.append(float(p_matrix[i, j, tau]))
        _, q_vals, _, _ = multipletests(np.array(flat_p), alpha=alpha, method="fdr_bh")
        for (i, j, tau), q in zip(flat_indices, q_vals):
            q_matrix[i, j, tau] = q

    rows: list[dict] = []
    n_vars = len(variables)
    for i, source in enumerate(variables):
        for j, target in enumerate(variables):
            if i == j:
                continue
            candidates = []
            for tau in range(1, max_lag + 1):
                candidates.append(
                    {
                        "lag": tau,
                        "score": abs(float(val_matrix[i, j, tau])),
                        "signed_score": float(val_matrix[i, j, tau]),
                        "p_value": float(p_matrix[i, j, tau]),
                        "q_value": float(q_matrix[i, j, tau]),
                    }
                )
            best = min(candidates, key=lambda row: (row["q_value"], -row["score"]))
            rows.append(
                {
                    "method": method_name,
                    "source": source,
                    "target": target,
                    "lag": int(best["lag"]),
                    "score": best["score"],
                    "signed_score": best["signed_score"],
                    "p_value": best["p_value"],
                    "q_value": best["q_value"],
                    "significant": bool(best["q_value"] <= alpha),
                    "pc_alpha": alpha if pc_alpha is None else pc_alpha,
                }
            )
    return pd.DataFrame(rows)

