"""Synthetic lagged causal benchmark data.

The benchmark uses six observed variables, X1-X6, and five complementary
scenarios. Each scenario mixes strong, medium, and weak directed edges so that
the benchmark retains useful variation at T=500.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


VARIABLES = tuple(f"X{i}" for i in range(1, 7))
DATASETS = (
    "S0_linear_mixed_strength",
    "S1_indirect_conditional",
    "S2_hidden_confounder",
    "S3_nonlinear_nonmonotonic",
    "S4_feedback_mixed_lag",
)

STRONG_EDGE_THRESHOLD = 0.35
MEDIUM_EDGE_THRESHOLD = 0.22


@dataclass(frozen=True)
class SyntheticResult:
    name: str
    data: pd.DataFrame
    truth: pd.DataFrame
    metadata: pd.DataFrame


def _standardize(arr: np.ndarray) -> np.ndarray:
    means = arr.mean(axis=0, keepdims=True)
    stds = arr.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    return (arr - means) / stds


def _strength_class(effect_size: float) -> str:
    magnitude = abs(float(effect_size))
    if magnitude >= STRONG_EDGE_THRESHOLD:
        return "strong"
    if magnitude >= MEDIUM_EDGE_THRESHOLD:
        return "medium"
    return "weak"


def _truth(dataset: str, rows: list[tuple[str, str, int, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "source": source,
                "target": target,
                "lag": lag,
                "relation": relation,
                "active_state": active_state,
                "effect_size": effect_size,
                "strength_class": _strength_class(effect_size),
            }
            for source, target, lag, relation, active_state, effect_size in rows
        ]
    )


def _finish(
    name: str,
    x: np.ndarray,
    burn_in: int,
    t: int,
    truth: pd.DataFrame,
    metadata_extra: dict[str, np.ndarray] | None = None,
) -> SyntheticResult:
    observed = _standardize(x[burn_in : burn_in + t])
    data = pd.DataFrame(observed, columns=VARIABLES)
    metadata = pd.DataFrame({"t": np.arange(t)})
    if metadata_extra:
        for key, values in metadata_extra.items():
            metadata[key] = values[burn_in : burn_in + t]
    return SyntheticResult(name=name, data=data, truth=truth, metadata=metadata)


def _noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.normal(scale=scale, size=(n, len(VARIABLES)))


def _even_saturation(value: float) -> float:
    """Bounded even transform with little linear correlation to its input."""
    return (np.tanh(value) ** 2 - 0.40) / 0.30


def _absolute_threshold(value: float) -> float:
    """Centered two-sided threshold response."""
    return (float(abs(value) > 0.80) - 0.42) / 0.49


def _odd_saturation(value: float) -> float:
    return np.tanh(1.20 * value) / 0.70


def generate_s0_linear_mixed_strength(
    t: int = 500,
    seed: int = 0,
    burn_in: int = 500,
) -> SyntheticResult:
    """Linear baseline with mixed signs, strengths, and lags 1-5."""
    name = "S0_linear_mixed_strength"
    rng = np.random.default_rng(seed)
    n = t + burn_in
    x = np.zeros((n, len(VARIABLES)))
    eps = _noise(rng, n, scale=1.00)

    for i in range(6, n):
        x[i, 0] = 0.42 * x[i - 1, 0] + eps[i, 0]
        x[i, 1] = 0.34 * x[i - 1, 1] + 0.48 * x[i - 1, 0] + eps[i, 1]
        x[i, 2] = 0.40 * x[i - 1, 2] + 0.30 * x[i - 3, 0] + eps[i, 2]
        x[i, 3] = 0.30 * x[i - 1, 3] - 0.34 * x[i - 2, 1] + 0.20 * x[i - 1, 2] + eps[i, 3]
        x[i, 4] = 0.38 * x[i - 1, 4] + 0.38 * x[i - 4, 3] + eps[i, 4]
        x[i, 5] = 0.25 * x[i - 1, 5] + 0.14 * x[i - 5, 1] + eps[i, 5]

    truth = _truth(
        name,
        [
            ("X1", "X2", 1, "linear", "all", 0.48),
            ("X1", "X3", 3, "linear", "all", 0.30),
            ("X2", "X4", 2, "linear_negative", "all", -0.34),
            ("X3", "X4", 1, "linear_weak", "all", 0.20),
            ("X4", "X5", 4, "linear", "all", 0.38),
            ("X2", "X6", 5, "linear_weak", "all", 0.14),
        ],
    )
    return _finish(name, x, burn_in, t, truth)


def generate_s1_indirect_conditional(
    t: int = 500,
    seed: int = 0,
    burn_in: int = 500,
) -> SyntheticResult:
    """Chains, an observed fork, and a collider with no shortcut edges."""
    name = "S1_indirect_conditional"
    rng = np.random.default_rng(seed)
    n = t + burn_in
    x = np.zeros((n, len(VARIABLES)))
    eps = _noise(rng, n, scale=0.95)

    for i in range(6, n):
        x[i, 0] = 0.45 * x[i - 1, 0] + eps[i, 0]
        x[i, 1] = 0.35 * x[i - 1, 1] + 0.45 * x[i - 1, 0] + eps[i, 1]
        x[i, 2] = 0.38 * x[i - 1, 2] + 0.28 * x[i - 2, 1] + eps[i, 2]
        x[i, 3] = 0.40 * x[i - 1, 3] - 0.32 * x[i - 2, 0] + eps[i, 3]
        x[i, 4] = 0.30 * x[i - 1, 4] + 0.26 * x[i - 1, 3] + eps[i, 4]
        x[i, 5] = 0.28 * x[i - 1, 5] + 0.16 * x[i - 4, 1] + 0.34 * x[i - 2, 4] + eps[i, 5]

    truth = _truth(
        name,
        [
            ("X1", "X2", 1, "chain_parent", "all", 0.45),
            ("X2", "X3", 2, "chain_child", "all", 0.28),
            ("X1", "X4", 2, "observed_fork", "all", -0.32),
            ("X4", "X5", 1, "chain_child", "all", 0.26),
            ("X2", "X6", 4, "collider_parent_weak", "all", 0.16),
            ("X5", "X6", 2, "collider_parent", "all", 0.34),
        ],
    )
    return _finish(name, x, burn_in, t, truth)


def generate_s2_hidden_confounder(
    t: int = 500,
    seed: int = 0,
    burn_in: int = 500,
) -> SyntheticResult:
    """A latent AR driver reaches X1 and X2 at different delays."""
    name = "S2_hidden_confounder"
    rng = np.random.default_rng(seed)
    n = t + burn_in
    x = np.zeros((n, len(VARIABLES)))
    u = np.zeros(n)
    eps = _noise(rng, n, scale=0.95)
    eps_u = rng.normal(scale=0.90, size=n)

    for i in range(6, n):
        u[i] = 0.70 * u[i - 1] + eps_u[i]
        x[i, 0] = 0.35 * x[i - 1, 0] + 0.60 * u[i - 1] + eps[i, 0]
        x[i, 1] = 0.40 * x[i - 1, 1] - 0.55 * u[i - 3] + eps[i, 1]
        x[i, 2] = 0.42 * x[i - 1, 2] + 0.30 * x[i - 2, 0] + eps[i, 2]
        x[i, 3] = 0.30 * x[i - 1, 3] + 0.27 * x[i - 1, 1] + eps[i, 3]
        x[i, 4] = 0.38 * x[i - 1, 4] - 0.18 * x[i - 4, 2] + eps[i, 4]
        x[i, 5] = 0.25 * x[i - 1, 5] + 0.35 * x[i - 2, 3] + eps[i, 5]

    truth = _truth(
        name,
        [
            ("X1", "X3", 2, "observed_linear", "all", 0.30),
            ("X2", "X4", 1, "observed_linear", "all", 0.27),
            ("X3", "X5", 4, "observed_linear_weak", "all", -0.18),
            ("X4", "X6", 2, "observed_linear", "all", 0.35),
        ],
    )
    hidden_u = _standardize(u[:, None]).ravel()
    return _finish(name, x, burn_in, t, truth, metadata_extra={"hidden_U": hidden_u})


def generate_s3_nonlinear_nonmonotonic(
    t: int = 500,
    seed: int = 0,
    burn_in: int = 500,
) -> SyntheticResult:
    """Mixed nonlinear graph containing even, threshold, and saturating links."""
    name = "S3_nonlinear_nonmonotonic"
    rng = np.random.default_rng(seed)
    n = t + burn_in
    x = np.zeros((n, len(VARIABLES)))
    eps = _noise(rng, n, scale=0.85)

    for i in range(6, n):
        x[i, 0] = 0.40 * x[i - 1, 0] + eps[i, 0]
        x[i, 1] = 0.28 * x[i - 1, 1] + 0.42 * _even_saturation(x[i - 1, 0]) + eps[i, 1]
        x[i, 2] = 0.32 * x[i - 1, 2] + 0.30 * _odd_saturation(x[i - 2, 1]) + eps[i, 2]
        x[i, 3] = 0.25 * x[i - 1, 3] + 0.38 * _absolute_threshold(x[i - 1, 2]) + eps[i, 3]
        x[i, 4] = 0.30 * x[i - 1, 4] + 0.20 * x[i - 3, 3] + eps[i, 4]
        x[i, 5] = (
            0.25 * x[i - 1, 5]
            + 0.35 * _even_saturation(x[i - 4, 0])
            + 0.28 * _odd_saturation(x[i - 2, 4])
            + eps[i, 5]
        )

    truth = _truth(
        name,
        [
            ("X1", "X2", 1, "nonlinear_even_saturation", "all", 0.42),
            ("X2", "X3", 2, "nonlinear_odd_saturation", "all", 0.30),
            ("X3", "X4", 1, "nonlinear_absolute_threshold", "all", 0.38),
            ("X4", "X5", 3, "linear_anchor_weak", "all", 0.20),
            ("X1", "X6", 4, "nonlinear_even_saturation", "all", 0.35),
            ("X5", "X6", 2, "nonlinear_odd_saturation", "all", 0.28),
        ],
    )
    return _finish(name, x, burn_in, t, truth)


def generate_s4_feedback_mixed_lag(
    t: int = 500,
    seed: int = 0,
    burn_in: int = 500,
) -> SyntheticResult:
    """Two asymmetric feedback pairs plus a long-lag transmission edge."""
    name = "S4_feedback_mixed_lag"
    rng = np.random.default_rng(seed)
    n = t + burn_in
    x = np.zeros((n, len(VARIABLES)))
    eps = _noise(rng, n, scale=0.95)

    for i in range(6, n):
        x[i, 0] = 0.35 * x[i - 1, 0] + 0.20 * x[i - 4, 1] + eps[i, 0]
        x[i, 1] = 0.32 * x[i - 1, 1] + 0.42 * x[i - 1, 0] + eps[i, 1]
        x[i, 2] = 0.40 * x[i - 1, 2] - 0.28 * x[i - 2, 1] + eps[i, 2]
        x[i, 3] = 0.30 * x[i - 1, 3] + 0.20 * x[i - 3, 4] + eps[i, 3]
        x[i, 4] = 0.35 * x[i - 1, 4] + 0.40 * x[i - 1, 3] + eps[i, 4]
        x[i, 5] = 0.25 * x[i - 1, 5] + 0.25 * x[i - 5, 2] + eps[i, 5]

    truth = _truth(
        name,
        [
            ("X1", "X2", 1, "feedback_forward_strong", "all", 0.42),
            ("X2", "X1", 4, "feedback_reverse_weak", "all", 0.20),
            ("X2", "X3", 2, "linear_negative", "all", -0.28),
            ("X4", "X5", 1, "feedback_forward_strong", "all", 0.40),
            ("X5", "X4", 3, "feedback_reverse_weak", "all", 0.20),
            ("X3", "X6", 5, "long_lag_linear", "all", 0.25),
        ],
    )
    return _finish(name, x, burn_in, t, truth)


GENERATORS: dict[str, Callable[[int, int, int], SyntheticResult]] = {
    "S0_LINEAR_MIXED_STRENGTH": generate_s0_linear_mixed_strength,
    "S1_INDIRECT_CONDITIONAL": generate_s1_indirect_conditional,
    "S2_HIDDEN_CONFOUNDER": generate_s2_hidden_confounder,
    "S3_NONLINEAR_NONMONOTONIC": generate_s3_nonlinear_nonmonotonic,
    "S4_FEEDBACK_MIXED_LAG": generate_s4_feedback_mixed_lag,
}


def generate_dataset(name: str, t: int = 500, seed: int = 0, burn_in: int = 500) -> SyntheticResult:
    key = name.upper()
    if key not in GENERATORS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(GENERATORS)}.")
    return GENERATORS[key](t=t, seed=seed, burn_in=burn_in)


def save_synthetic_panel(output_dir: str | Path, t: int = 500, seeds: int = 10, burn_in: int = 500) -> pd.DataFrame:
    """Generate all S0-S4 datasets for seed values 0..seeds-1."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth_frames: list[pd.DataFrame] = []
    for seed in range(seeds):
        for dataset in DATASETS:
            result = generate_dataset(dataset, t=t, seed=seed, burn_in=burn_in)
            result.data.to_csv(out / f"dataset_{dataset}_seed_{seed:02d}.csv", index=False)
            result.metadata.to_csv(out / f"metadata_{dataset}_seed_{seed:02d}.csv", index=False)
            truth_frames.append(result.truth.assign(seed=seed))
    truth = pd.concat(truth_frames, ignore_index=True)
    truth.to_csv(out / "ground_truth.csv", index=False)
    return truth
