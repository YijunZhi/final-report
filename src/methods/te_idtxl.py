"""Formal Transfer Entropy using IDTxl MultivariateTE + JidtKraskovCMI."""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_NAME = "TE-IDTxl-MultivariateTE"
DEFAULT_PARAMS = {
    "implementation": "IDTxl.MultivariateTE",
    "cmi_estimator": "JidtKraskovCMI",
    "max_lag_sources": 5,
    "min_lag_sources": 1,
    "max_lag_target": 5,
    "tau_sources": 1,
    "tau_target": 1,
    "n_perm_max_stat": 25,
    "n_perm_min_stat": 25,
    "n_perm_omnibus": 25,
    "n_perm_max_seq": 25,
    "alpha": 0.05,
    "kraskov_k": 4,
    "theiler_t": 0,
    "noise_level": 1e-8,
    "permute_in_time": True,
    "fdr_correction": False,
}


def _patch_numpy_for_idtxl() -> None:
    if not hasattr(np, "math"):
        np.math = math  # type: ignore[attr-defined]
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]


def _configure_java_home() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    env_root = Path(conda_prefix)
    java_home = env_root / "Library" / "lib" / "jvm"
    java_bin = java_home / "bin"
    library_bin = env_root / "Library" / "bin"
    if java_home.exists():
        os.environ.setdefault("JAVA_HOME", str(java_home))
        os.environ["PATH"] = os.pathsep.join([str(java_bin), str(library_bin), os.environ.get("PATH", "")])


def _start_jvm_with_extracted_jidt() -> None:
    _patch_numpy_for_idtxl()
    _configure_java_home()
    import idtxl
    import jpype as jp

    if jp.isJVMStarted():
        return

    jar_path = Path(idtxl.__file__).with_name("infodynamics.jar")
    class_dir = Path(tempfile.gettempdir()) / "idtxl_infodynamics_classes"
    marker = (
        class_dir
        / "infodynamics"
        / "measures"
        / "continuous"
        / "kraskov"
        / "ConditionalMutualInfoCalculatorMultiVariateKraskov1.class"
    )
    if not marker.exists():
        class_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(jar_path) as zf:
            zf.extractall(class_dir)

    jp.startJVM(jp.getDefaultJVMPath(), "-ea", f"-Djava.class.path={class_dir}")


def _to_idtxl_data(frame: pd.DataFrame):
    from idtxl.data import Data

    data = Data(normalise=False)
    data.set_data(frame.to_numpy(dtype=float).T, "ps")
    return data


def _settings(max_lag: int, alpha: float, n_perm: int, kraskov_k: int, fdr_correction: bool, verbose: bool) -> dict:
    if n_perm <= int(1 / alpha):
        raise ValueError("IDTxl requires n_perm to be greater than 1 / alpha.")
    return {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": int(max_lag),
        "min_lag_sources": 1,
        "max_lag_target": int(max_lag),
        "tau_sources": 1,
        "tau_target": 1,
        "n_perm_max_stat": int(n_perm),
        "n_perm_min_stat": int(n_perm),
        "n_perm_omnibus": int(n_perm),
        "n_perm_max_seq": int(n_perm),
        "alpha_max_stat": float(alpha),
        "alpha_min_stat": float(alpha),
        "alpha_omnibus": float(alpha),
        "alpha_max_seq": float(alpha),
        "permute_in_time": True,
        "fdr_correction": bool(fdr_correction),
        "verbose": bool(verbose),
        "kraskov_k": int(kraskov_k),
        "theiler_t": 0,
        "noise_level": 1e-8,
        "num_threads": 1,
    }


def _extract_edges(results, variables: list[str], alpha: float, use_fdr: bool) -> pd.DataFrame:
    rows: list[dict] = []
    n_vars = len(variables)
    for target_idx, target in enumerate(variables):
        try:
            single = results.get_single_target(target_idx, fdr=use_fdr)
        except Exception:
            single = None
        selected = [] if single is None else list(single.get("selected_vars_sources", []) or [])
        p_values = [] if single is None or single.get("selected_sources_pval") is None else list(single["selected_sources_pval"])
        te_values = [] if single is None or single.get("selected_sources_te") is None else list(single["selected_sources_te"])

        grouped: dict[int, list[dict]] = {i: [] for i in range(n_vars) if i != target_idx}
        for idx, source_lag in enumerate(selected):
            source_idx = int(source_lag[0])
            lag = int(source_lag[1])
            p_value = float(p_values[idx]) if idx < len(p_values) else np.nan
            te_value = float(te_values[idx]) if idx < len(te_values) else np.nan
            if source_idx != target_idx:
                grouped.setdefault(source_idx, []).append({"lag": lag, "p_value": p_value, "score": te_value})

        for source_idx, source in enumerate(variables):
            if source_idx == target_idx:
                continue
            candidates = grouped.get(source_idx, [])
            if candidates:
                best = sorted(
                    candidates,
                    key=lambda row: (
                        np.inf if np.isnan(row["p_value"]) else row["p_value"],
                        -(0.0 if np.isnan(row["score"]) else row["score"]),
                    ),
                )[0]
                rows.append(
                    {
                        "method": METHOD_NAME,
                        "source": source,
                        "target": target,
                        "lag": int(best["lag"]),
                        "score": best["score"],
                        "p_value": best["p_value"],
                        "q_value": best["p_value"],
                        "significant": bool(np.isnan(best["p_value"]) or best["p_value"] <= alpha),
                        "fdr_correction": use_fdr,
                    }
                )
            else:
                rows.append(
                    {
                        "method": METHOD_NAME,
                        "source": source,
                        "target": target,
                        "lag": np.nan,
                        "score": np.nan,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "significant": False,
                        "fdr_correction": use_fdr,
                    }
                )
    return pd.DataFrame(rows)


def infer_edges(
    data: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
    n_perm: int = 25,
    kraskov_k: int = 4,
    fdr_correction: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    frame = data.apply(pd.to_numeric, errors="coerce").dropna().astype(float)
    variables = list(frame.columns)
    settings = _settings(max_lag=max_lag, alpha=alpha, n_perm=n_perm, kraskov_k=kraskov_k, fdr_correction=fdr_correction, verbose=verbose)

    def _run():
        _start_jvm_with_extracted_jidt()
        from idtxl.multivariate_te import MultivariateTE

        idtxl_data = _to_idtxl_data(frame)
        sources = [[idx for idx in range(len(variables)) if idx != target] for target in range(len(variables))]
        return MultivariateTE().analyse_network(settings, idtxl_data, targets=list(range(len(variables))), sources=sources)

    if verbose:
        results = _run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            results = _run()

    return _extract_edges(results, variables=variables, alpha=alpha, use_fdr=fdr_correction)
