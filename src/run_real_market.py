"""Run formal causal discovery methods on the real market data file.

This runner is intentionally separate from the synthetic benchmark runner
because real market data has no known ground-truth edge set. It therefore
reports discovered edges, method consensus, and network summaries instead of
Precision/Recall/F1.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_LAG,
    METHOD_MODULES,
    REAL_MARKET_INPUT_CSV,
    REAL_MARKET_METHODS,
    REAL_MARKET_RESULTS_XLSX,
    REAL_MARKET_TE_N_PERM,
    TE_FDR_CORRECTION,
)
from metrics import edge_score_frame

DEFAULT_DATA = str(REAL_MARKET_INPUT_CSV)
DEFAULT_OUT = str(REAL_MARKET_RESULTS_XLSX)


def load_method(key: str):
    if key not in METHOD_MODULES:
        raise ValueError(f"Unknown method {key!r}; choose from {sorted(METHOD_MODULES)}.")
    module = importlib.import_module(METHOD_MODULES[key])
    return module.METHOD_NAME, module.DEFAULT_PARAMS, module.infer_edges


def load_real_market_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    variables = [col for col in raw.columns if col != "date"]
    data = raw[variables].apply(pd.to_numeric, errors="coerce")
    keep = data.notna().all(axis=1)
    cleaned = data.loc[keep].astype(float).reset_index(drop=True)
    cleaned_dates = raw.loc[keep, ["date"]].reset_index(drop=True)
    return cleaned_dates, cleaned


def data_summary_frame(path: Path, dates: pd.DataFrame, data: pd.DataFrame, raw_rows: int) -> pd.DataFrame:
    rows = [
        {"item": "source_file", "value": str(path)},
        {"item": "raw_rows", "value": raw_rows},
        {"item": "usable_rows_after_dropna", "value": len(data)},
        {"item": "date_min", "value": dates["date"].min().strftime("%Y-%m-%d")},
        {"item": "date_max", "value": dates["date"].max().strftime("%Y-%m-%d")},
        {"item": "variables", "value": ",".join(data.columns)},
    ]
    for col in data.columns:
        rows.extend(
            [
                {"item": f"{col}.mean", "value": float(data[col].mean())},
                {"item": f"{col}.std", "value": float(data[col].std())},
                {"item": f"{col}.min", "value": float(data[col].min())},
                {"item": f"{col}.max", "value": float(data[col].max())},
            ]
        )
    return pd.DataFrame(rows)


def method_parameters_frame(method_keys: list[str], args: argparse.Namespace, variables: list[str]) -> pd.DataFrame:
    rows: list[dict] = [
        {"method": "GLOBAL", "parameter": "input_file", "value": args.input},
        {"method": "GLOBAL", "parameter": "variables", "value": ",".join(variables)},
        {"method": "GLOBAL", "parameter": "max_lag", "value": args.max_lag},
        {"method": "GLOBAL", "parameter": "alpha", "value": args.alpha},
        {"method": "GLOBAL", "parameter": "methods", "value": ",".join(method_keys)},
        {"method": "GLOBAL", "parameter": "excluded_method", "value": "PCMCI-CMIknn"},
    ]
    for key in method_keys:
        method_name, params, _ = load_method(key)
        for name, value in params.items():
            rows.append({"method": method_name, "parameter": name, "value": value})
    if "te" in method_keys:
        rows.append({"method": "TE-IDTxl-MultivariateTE", "parameter": "runtime_override.n_perm", "value": args.te_n_perm})
        rows.append({"method": "TE-IDTxl-MultivariateTE", "parameter": "runtime_override.fdr_correction", "value": args.te_fdr})
    return pd.DataFrame(rows)


def infer_with_runtime(method_key: str, data: pd.DataFrame, args: argparse.Namespace) -> tuple[str, pd.DataFrame, float]:
    method_name, _, infer_edges = load_method(method_key)
    started = time.time()
    if method_key == "te":
        edges = infer_edges(
            data,
            max_lag=args.max_lag,
            alpha=args.alpha,
            n_perm=args.te_n_perm,
            fdr_correction=args.te_fdr,
        )
    else:
        edges = infer_edges(data, max_lag=args.max_lag, alpha=args.alpha)
    elapsed = time.time() - started
    return method_name, edges, elapsed


def consensus_edges(edge_results: pd.DataFrame) -> pd.DataFrame:
    if edge_results.empty:
        return pd.DataFrame()
    sig = edge_results[edge_results["significant"].astype(bool)].copy()
    if sig.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "method_count",
                "methods",
                "lags",
                "mean_lag",
                "mean_score",
                "mean_q_value",
            ]
        )
    out = (
        sig.groupby(["source", "target"], as_index=False)
        .agg(
            method_count=("method", "nunique"),
            methods=("method", lambda values: ", ".join(sorted(set(values)))),
            lags=("lag", lambda values: ", ".join(str(int(v)) for v in values if pd.notna(v))),
            mean_lag=("lag", "mean"),
            mean_score=("score", "mean"),
            mean_q_value=("q_value", "mean"),
        )
        .sort_values(["method_count", "source", "target"], ascending=[False, True, True])
    )
    return out


def network_summary(edge_results: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if edge_results.empty:
        return pd.DataFrame(columns=["method", "variable", "out_degree", "in_degree", "net_out_degree"])
    sig = edge_results[edge_results["significant"].astype(bool)].copy()
    methods = sorted(edge_results["method"].dropna().unique())
    for method in methods:
        sub = sig[sig["method"] == method]
        for variable in variables:
            out_degree = int((sub["source"] == variable).sum())
            in_degree = int((sub["target"] == variable).sum())
            rows.append(
                {
                    "method": method,
                    "variable": variable,
                    "out_degree": out_degree,
                    "in_degree": in_degree,
                    "net_out_degree": out_degree - in_degree,
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "net_out_degree"], ascending=[True, False])


def real_driver_scores(edge_results: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    empty_truth = pd.DataFrame(columns=["source", "target"])
    for method in sorted(edge_results["method"].dropna().unique()):
        sub = edge_results[edge_results["method"] == method].copy()
        scores = edge_score_frame(sub, empty_truth, variables)
        driver = (
            scores.groupby("source", as_index=False)
            .agg(
                estimated_driver_p_score=("p_score", "sum"),
                estimated_driver_effect_score=("effect_score", "sum"),
            )
            .rename(columns={"source": "variable"})
        )
        driver["rank_p_score"] = driver["estimated_driver_p_score"].rank(ascending=False, method="min")
        driver["rank_effect_score"] = driver["estimated_driver_effect_score"].rank(ascending=False, method="min")
        rows.append(driver.assign(method=method))
    if not rows:
        return pd.DataFrame(
            columns=[
                "variable",
                "estimated_driver_p_score",
                "estimated_driver_effect_score",
                "rank_p_score",
                "rank_effect_score",
                "method",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["method", "rank_p_score", "variable"])


def run_real_market(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    input_path = Path(args.input)
    raw_rows = len(pd.read_csv(input_path, usecols=["date"]))
    dates, data = load_real_market_data(input_path)
    variables = list(data.columns)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_edges: list[pd.DataFrame] = []
    runtime_rows: list[dict] = []
    started = time.time()

    for method_key in args.methods:
        method_name, edges, elapsed = infer_with_runtime(method_key, data, args)
        all_edges.append(edges)
        runtime_rows.append({"method": method_name, "seconds": elapsed})
        print(f"finished real_market method={method_name} seconds={elapsed:.1f}", flush=True)

    edge_results = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    significant = edge_results[edge_results["significant"].astype(bool)].copy() if not edge_results.empty else pd.DataFrame()
    summary = data_summary_frame(input_path, dates, data, raw_rows=raw_rows)
    params = method_parameters_frame(args.methods, args, variables)
    params.loc[len(params)] = {"method": "GLOBAL", "parameter": "total_elapsed_seconds", "value": round(time.time() - started, 3)}
    runtime = pd.DataFrame(runtime_rows)
    consensus = consensus_edges(edge_results)
    network = network_summary(edge_results, variables)
    driver_scores = real_driver_scores(edge_results, variables)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="data_summary", index=False)
        params.to_excel(writer, sheet_name="method_parameters", index=False)
        edge_results.to_excel(writer, sheet_name="all_edges", index=False)
        significant.to_excel(writer, sheet_name="significant_edges", index=False)
        consensus.to_excel(writer, sheet_name="consensus_edges", index=False)
        network.to_excel(writer, sheet_name="network_summary", index=False)
        driver_scores.to_excel(writer, sheet_name="driver_scores", index=False)
        runtime.to_excel(writer, sheet_name="runtime_log", index=False)

    return {
        "data_summary": summary,
        "method_parameters": params,
        "all_edges": edge_results,
        "significant_edges": significant,
        "consensus_edges": consensus,
        "network_summary": network,
        "driver_scores": driver_scores,
        "runtime_log": runtime,
    }


def dry_run(args: argparse.Namespace) -> None:
    dates, data = load_real_market_data(args.input)
    print("Real market dry run")
    print(f"input={args.input}")
    print(f"shape={data.shape}")
    print(f"date_range={dates['date'].min().date()}..{dates['date'].max().date()}")
    print(f"variables={','.join(data.columns)}")
    print(f"max_lag={args.max_lag}, alpha={args.alpha}")
    print(f"methods={','.join(args.methods)}")
    for key in args.methods:
        method_name, params, _ = load_method(key)
        print(f"\n[{method_name}]")
        print(json.dumps(params, ensure_ascii=False, indent=2, default=str))
    print("\nDry run complete: no methods were executed and no result files were written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--methods", nargs="+", choices=sorted(REAL_MARKET_METHODS), default=list(REAL_MARKET_METHODS))
    parser.add_argument("--te-n-perm", type=int, default=REAL_MARKET_TE_N_PERM)
    parser.add_argument("--te-fdr", action="store_true", default=TE_FDR_CORRECTION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.dry_run:
        dry_run(parsed)
    else:
        run_real_market(parsed)
