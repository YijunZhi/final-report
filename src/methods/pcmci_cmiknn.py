"""Formal PCMCI-CMIknn method using tigramite.

This is the real tigramite PCMCI + CMIknn workflow.  It is much slower than
PCMCI-ParCorr because it uses k-nearest-neighbour conditional mutual information
with shuffle significance tests.
"""

from __future__ import annotations

import pandas as pd
from tigramite.independence_tests.cmiknn import CMIknn

from methods.pcmci_utils import run_pcmci_edges


METHOD_NAME = "PCMCI-CMIknn"
DEFAULT_PARAMS = {
    "implementation": "tigramite.PCMCI",
    "conditional_independence_test": "CMIknn",
    "tau_min": 1,
    "tau_max": 5,
    "knn": 0.2,
    "shuffle_neighbors": 5,
    "significance": "shuffle_test",
    "sig_samples": 100,
    "sig_blocklength": 5,
    "transform": "ranks",
    "workers": -1,
    "pc_alpha": 0.05,
    "alpha": 0.05,
    "multiple_testing": "tigramite FDR-BH corrected p-values",
}


def infer_edges(
    data: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
    pc_alpha: float | None = None,
    knn: float = 0.2,
    shuffle_neighbors: int = 5,
    sig_samples: int = 100,
    sig_blocklength: int = 5,
    transform: str = "ranks",
    workers: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    cond_ind_test = CMIknn(
        knn=knn,
        shuffle_neighbors=shuffle_neighbors,
        significance="shuffle_test",
        transform=transform,
        sig_samples=sig_samples,
        sig_blocklength=sig_blocklength,
        workers=workers,
        seed=seed,
    )
    return run_pcmci_edges(
        data=data,
        cond_ind_test=cond_ind_test,
        method_name=METHOD_NAME,
        max_lag=max_lag,
        alpha=alpha,
        pc_alpha=pc_alpha,
    )

