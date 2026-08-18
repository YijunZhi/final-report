"""Evaluation metrics for directed synthetic causal edge recovery."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _edge_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.empty:
        return set()
    return set(zip(frame["source"].astype(str), frame["target"].astype(str)))


def evaluate_edges(predicted: pd.DataFrame, truth: pd.DataFrame, variables: list[str]) -> dict:
    pred_sig = predicted[predicted["significant"].astype(bool)].copy() if not predicted.empty else predicted.copy()
    pred_edges = _edge_set(pred_sig)
    truth_edges = _edge_set(truth)
    tp = len(pred_edges & truth_edges)
    fp = len(pred_edges - truth_edges)
    fn = len(truth_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    shd = fp + fn

    truth_unordered = {frozenset(edge) for edge in truth_edges}
    pred_on_truth_pairs = [edge for edge in pred_edges if frozenset(edge) in truth_unordered]
    direction_accuracy = tp / len(pred_on_truth_pairs) if pred_on_truth_pairs else np.nan

    truth_lags = {(row.source, row.target): int(row.lag) for row in truth.itertuples(index=False)}
    lag_errors = []
    exact_lag_hits = 0
    if not pred_sig.empty:
        for row in pred_sig.itertuples(index=False):
            key = (str(row.source), str(row.target))
            if key in truth_lags and not pd.isna(row.lag):
                lag_error = abs(int(row.lag) - truth_lags[key])
                lag_errors.append(lag_error)
                exact_lag_hits += int(lag_error == 0)
    mean_lag_error = float(np.mean(lag_errors)) if lag_errors else np.nan
    exact_lag_accuracy = exact_lag_hits / len(lag_errors) if lag_errors else np.nan

    truth_with_strength = truth.copy()
    if "strength_class" not in truth_with_strength.columns:
        if "effect_size" in truth_with_strength.columns:
            magnitudes = pd.to_numeric(truth_with_strength["effect_size"], errors="coerce").abs()
        else:
            magnitudes = pd.Series(1.0, index=truth_with_strength.index)
        truth_with_strength["strength_class"] = np.select(
            [magnitudes >= 0.35, magnitudes >= 0.22],
            ["strong", "medium"],
            default="weak",
        )
    strength_recalls: dict[str, float] = {}
    for strength in ("strong", "medium", "weak"):
        strength_truth = _edge_set(truth_with_strength[truth_with_strength["strength_class"] == strength])
        strength_recalls[f"{strength}_edge_recall"] = (
            len(pred_edges & strength_truth) / len(strength_truth) if strength_truth else np.nan
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "direction_accuracy": direction_accuracy,
        "mean_lag_error": mean_lag_error,
        "exact_lag_accuracy": exact_lag_accuracy,
        **strength_recalls,
        "n_pred_edges": len(pred_edges),
        "n_true_edges": len(truth_edges),
        "n_possible_edges": len(list(itertools.permutations(variables, 2))),
    }


def _candidate_pairs(variables: list[str]) -> list[tuple[str, str]]:
    return [(str(source), str(target)) for source, target in itertools.permutations(variables, 2)]


def _safe_neg_log10(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    scores = np.zeros_like(numeric, dtype=float)
    mask = np.isfinite(numeric)
    clipped = np.clip(numeric[mask], 1e-300, 1.0)
    scores[mask] = -np.log10(clipped)
    return scores


def _effect_score_series(frame: pd.DataFrame) -> pd.Series:
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in ("effect_score", "lag_score", "score"):
        if column in frame.columns:
            scores = scores.fillna(pd.to_numeric(frame[column], errors="coerce").abs())
    return scores.fillna(0.0)


def edge_score_frame(predicted: pd.DataFrame, truth: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Build one continuous p-score/effect-score row for every ordered pair."""
    pairs = _candidate_pairs(variables)
    out = pd.DataFrame(pairs, columns=["source", "target"])
    truth_edges = _edge_set(truth)
    out["is_true_edge"] = [int((source, target) in truth_edges) for source, target in pairs]
    out["p_score"] = 0.0
    out["effect_score"] = 0.0

    if predicted.empty:
        return out

    scored = predicted.copy()
    p_values = pd.Series(np.nan, index=scored.index, dtype=float)
    if "q_value" in scored.columns:
        p_values = pd.to_numeric(scored["q_value"], errors="coerce")
    if "p_value" in scored.columns:
        p_values = p_values.fillna(pd.to_numeric(scored["p_value"], errors="coerce"))
    scored["_p_score_eval"] = _safe_neg_log10(p_values)
    scored["_effect_score_eval"] = _effect_score_series(scored).fillna(0.0)

    grouped = (
        scored.groupby(["source", "target"], as_index=False)
        .agg(p_score=("_p_score_eval", "max"), effect_score=("_effect_score_eval", "max"))
    )
    out = out.drop(columns=["p_score", "effect_score"]).merge(grouped, on=["source", "target"], how="left")
    out[["p_score", "effect_score"]] = out[["p_score", "effect_score"]].fillna(0.0)
    return out


def edge_ranking_metrics(predicted: pd.DataFrame, truth: pd.DataFrame, variables: list[str]) -> dict:
    """Evaluate threshold-free edge ranking by AUROC and AUPRC."""
    scores = edge_score_frame(predicted, truth, variables)
    y_true = scores["is_true_edge"].to_numpy(dtype=int)
    out: dict[str, float] = {}
    for score_col, suffix in [("p_score", "p_score"), ("effect_score", "effect_score")]:
        y_score = scores[score_col].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            out[f"edge_auroc_{suffix}"] = np.nan
            out[f"edge_auprc_{suffix}"] = np.nan
        else:
            out[f"edge_auroc_{suffix}"] = float(roc_auc_score(y_true, y_score))
            out[f"edge_auprc_{suffix}"] = float(average_precision_score(y_true, y_score))
    return out


def driver_score_table(predicted: pd.DataFrame, truth: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Compare true and estimated source-level driver strength."""
    variables = [str(variable) for variable in variables]
    edge_scores = edge_score_frame(predicted, truth, variables)
    estimated = (
        edge_scores.groupby("source", as_index=False)
        .agg(
            estimated_driver_p_score=("p_score", "sum"),
            estimated_driver_effect_score=("effect_score", "sum"),
        )
        .rename(columns={"source": "variable"})
    )

    truth_weights = truth.copy()
    if truth_weights.empty:
        true_driver = pd.DataFrame({"variable": variables, "true_driver_score": 0.0})
    else:
        if "effect_size" in truth_weights.columns:
            truth_weights["_truth_weight"] = pd.to_numeric(truth_weights["effect_size"], errors="coerce").abs().fillna(0.0)
        else:
            truth_weights["_truth_weight"] = 1.0
        true_driver = (
            truth_weights.groupby("source", as_index=False)
            .agg(true_driver_score=("_truth_weight", "sum"))
            .rename(columns={"source": "variable"})
        )

    table = pd.DataFrame({"variable": variables})
    table = table.merge(true_driver, on="variable", how="left")
    table = table.merge(estimated, on="variable", how="left")
    for column in ["true_driver_score", "estimated_driver_p_score", "estimated_driver_effect_score"]:
        table[column] = table[column].fillna(0.0)
    table["true_rank"] = table["true_driver_score"].rank(ascending=False, method="min")
    table["estimated_rank_p_score"] = table["estimated_driver_p_score"].rank(ascending=False, method="min")
    table["estimated_rank_effect_score"] = table["estimated_driver_effect_score"].rank(ascending=False, method="min")
    return table.sort_values(["true_rank", "variable"]).reset_index(drop=True)


def _spearman_safe(a: pd.Series, b: pd.Series) -> float:
    left = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2 or len(np.unique(left[mask])) < 2 or len(np.unique(right[mask])) < 2:
        return np.nan
    return float(spearmanr(left[mask], right[mask]).statistic)


def _top_driver_set(table: pd.DataFrame, score_col: str, k: int, positive_only: bool = False) -> set[str]:
    ranked = table[["variable", score_col]].copy()
    if positive_only:
        ranked = ranked[ranked[score_col] > 0]
    if ranked.empty:
        return set()
    ranked = ranked.sort_values([score_col, "variable"], ascending=[False, True])
    return set(ranked.head(min(k, len(ranked)))["variable"].astype(str))


def driver_ranking_metrics(predicted: pd.DataFrame, truth: pd.DataFrame, variables: list[str], top_ks: tuple[int, ...] = (2, 3)) -> dict:
    table = driver_score_table(predicted, truth, variables)
    out = {
        "spearman_driver_p_score": _spearman_safe(table["true_driver_score"], table["estimated_driver_p_score"]),
        "spearman_driver_effect_score": _spearman_safe(table["true_driver_score"], table["estimated_driver_effect_score"]),
    }
    for k in top_ks:
        true_top = _top_driver_set(table, "true_driver_score", k, positive_only=True)
        for score_col, suffix in [
            ("estimated_driver_p_score", "p_score"),
            ("estimated_driver_effect_score", "effect_score"),
        ]:
            estimated_top = _top_driver_set(table, score_col, k, positive_only=False)
            out[f"top{k}_driver_recall_{suffix}"] = len(true_top & estimated_top) / len(true_top) if true_top else np.nan
    return out


def ranking_metrics(predicted: pd.DataFrame, truth: pd.DataFrame, variables: list[str]) -> dict:
    out = edge_ranking_metrics(predicted, truth, variables)
    out.update(driver_ranking_metrics(predicted, truth, variables))
    return out


def summarize_metrics(metrics_by_run: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "precision",
        "recall",
        "f1",
        "shd",
        "direction_accuracy",
        "mean_lag_error",
        "exact_lag_accuracy",
        "strong_edge_recall",
        "medium_edge_recall",
        "weak_edge_recall",
        "n_pred_edges",
    ]
    summary = metrics_by_run.groupby(["dataset", "method"])[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def summarize_ranking_metrics(ranking_by_run: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        col
        for col in ranking_by_run.columns
        if col not in {"dataset", "seed", "method"}
    ]
    summary = ranking_by_run.groupby(["dataset", "method"])[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def edge_stability(edge_results: pd.DataFrame, seeds: int) -> pd.DataFrame:
    if edge_results.empty:
        return pd.DataFrame(columns=["dataset", "method", "source", "target", "found_count", "stability"])
    sig = edge_results[edge_results["significant"].astype(bool)].copy()
    if sig.empty:
        return pd.DataFrame(columns=["dataset", "method", "source", "target", "found_count", "stability"])
    out = (
        sig.groupby(["dataset", "method", "source", "target"], as_index=False)
        .agg(found_count=("seed", "nunique"), mean_lag=("lag", "mean"), mean_score=("score", "mean"))
    )
    out["stability"] = out["found_count"] / seeds
    return out.sort_values(["dataset", "method", "stability", "source", "target"], ascending=[True, True, False, True, True])
