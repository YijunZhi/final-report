"""Formal PCMCI-ParCorr method using tigramite."""

from __future__ import annotations

import pandas as pd
from tigramite.independence_tests.parcorr import ParCorr

from methods.pcmci_utils import run_pcmci_edges


METHOD_NAME = "PCMCI-ParCorr"
DEFAULT_PARAMS = {
    "implementation": "tigramite.PCMCI",
    "conditional_independence_test": "ParCorr(significance='analytic')",
    "tau_min": 1,
    "tau_max": 5,
    "pc_alpha": 0.05,
    "alpha": 0.05,
    "multiple_testing": "tigramite FDR-BH corrected p-values",
}


def infer_edges(data: pd.DataFrame, max_lag: int = 5, alpha: float = 0.05, pc_alpha: float | None = None) -> pd.DataFrame:
    cond_ind_test = ParCorr(significance="analytic")
    return run_pcmci_edges(
        data=data,
        cond_ind_test=cond_ind_test,
        method_name=METHOD_NAME,
        max_lag=max_lag,
        alpha=alpha,
        pc_alpha=pc_alpha,
    )

