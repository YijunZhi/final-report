"""Run the final synthetic causal benchmark from data generation to metrics.

This runner is the canonical synthetic workflow for the thesis. It generates
the S0-S4 datasets, runs the selected formal causal discovery methods, computes
both binary edge-recovery metrics and ranking-based diagnostics, and writes a
single Excel workbook.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from config import (
    CMIKNN_SIG_BLOCKLENGTH,
    CMIKNN_SIG_SAMPLES,
    CMIKNN_WORKERS,
    DEFAULT_ALPHA,
    DEFAULT_MAX_LAG,
    DEFAULT_SEEDS,
    DEFAULT_T,
    FINAL_SYNTHETIC_DATA_DIR,
    METHOD_MODULES,
    RUN_LOG_DIR,
    SYNTHETIC_METHODS,
    SYNTHETIC_RESULTS_XLSX,
    SYNTHETIC_TE_N_PERM,
    TE_FDR_CORRECTION,
)
from metrics import (
    driver_score_table,
    edge_stability,
    evaluate_edges,
    ranking_metrics,
    summarize_metrics,
    summarize_ranking_metrics,
)
from synthetic_data import DATASETS, VARIABLES, generate_dataset


def load_method(key: str):
    if key not in METHOD_MODULES:
        raise ValueError(f"Unknown method {key!r}; choose from {sorted(METHOD_MODULES)}.")
    module = importlib.import_module(METHOD_MODULES[key])
    return module.METHOD_NAME, module.DEFAULT_PARAMS, module.infer_edges


def method_parameters_frame(method_keys: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    general = {
        "T": args.t,
        "seeds_count": args.seeds,
        "seed_values": f"0..{args.seeds - 1}",
        "datasets": ",".join(DATASETS),
        "variables": ",".join(VARIABLES),
        "max_lag": args.max_lag,
        "alpha": args.alpha,
        "methods": ",".join(method_keys),
    }
    for name, value in general.items():
        rows.append({"method": "GLOBAL", "parameter": name, "value": value})

    for key in method_keys:
        method_name, params, _ = load_method(key)
        for name, value in params.items():
            rows.append({"method": method_name, "parameter": name, "value": value})

    overrides = {
        "PCMCI-CMIknn": {
            "sig_samples": args.cmiknn_sig_samples,
            "sig_blocklength": args.cmiknn_sig_blocklength,
            "workers": args.cmiknn_workers,
        },
        "TE-IDTxl-MultivariateTE": {
            "n_perm": args.te_n_perm,
            "fdr_correction": args.te_fdr,
        },
    }
    loaded_names = {load_method(key)[0] for key in method_keys}
    for method_name, params in overrides.items():
        if method_name in loaded_names:
            for name, value in params.items():
                rows.append({"method": method_name, "parameter": f"runtime_override.{name}", "value": value})
    return pd.DataFrame(rows)


def dry_run(args: argparse.Namespace) -> None:
    print("Final synthetic benchmark dry run")
    print(f"T={args.t}, seeds=0..{args.seeds - 1}, max_lag={args.max_lag}, alpha={args.alpha}")
    print(f"datasets={','.join(DATASETS)}")
    print(f"variables={','.join(VARIABLES)}")
    print(f"methods={','.join(args.methods)}")
    print(f"output={args.out}")
    for key in args.methods:
        method_name, params, _ = load_method(key)
        print(f"\n[{method_name}]")
        print(json.dumps(params, ensure_ascii=False, indent=2, default=str))

    sample = generate_dataset(DATASETS[0], t=20, seed=0)
    print(f"\nSynthetic generator OK: shape={sample.data.shape}, truth_edges={len(sample.truth)}")
    print("Dry run complete: no methods were executed and no result files were written.")


def run_method(method_key: str, infer_edges: Callable, data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if method_key == "pcmci_cmiknn":
        return infer_edges(
            data,
            max_lag=args.max_lag,
            alpha=args.alpha,
            sig_samples=args.cmiknn_sig_samples,
            sig_blocklength=args.cmiknn_sig_blocklength,
            workers=args.cmiknn_workers,
        )
    if method_key == "te":
        return infer_edges(
            data,
            max_lag=args.max_lag,
            alpha=args.alpha,
            n_perm=args.te_n_perm,
            fdr_correction=args.te_fdr,
        )
    return infer_edges(data, max_lag=args.max_lag, alpha=args.alpha)


def run_synthetic_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    method_keys = args.methods
    loaded_methods = [(key, *load_method(key)) for key in method_keys]
    out = Path(args.out)
    data_dir = Path(args.data_dir)
    log_dir = Path(args.log_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    ranking_rows: list[dict] = []
    driver_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict] = []
    started = time.time()

    for seed in range(args.seeds):
        for dataset in DATASETS:
            result = generate_dataset(dataset, t=args.t, seed=seed)
            result.data.to_csv(data_dir / f"dataset_{dataset}_seed_{seed:02d}.csv", index=False)
            result.metadata.assign(dataset=dataset, seed=seed).to_csv(
                data_dir / f"metadata_{dataset}_seed_{seed:02d}.csv",
                index=False,
            )
            truth = result.truth.assign(seed=seed)
            truth_frames.append(truth)

            for method_key, method_name, _, infer_edges in loaded_methods:
                method_start = time.time()
                edges = run_method(method_key, infer_edges, result.data, args)
                elapsed = time.time() - method_start
                tagged = edges.assign(dataset=dataset, seed=seed)
                edge_frames.append(tagged)

                binary = evaluate_edges(tagged, truth, variables=list(VARIABLES))
                metrics_rows.append({"dataset": dataset, "seed": seed, "method": method_name, **binary})

                rank = ranking_metrics(tagged, truth, variables=list(VARIABLES))
                ranking_rows.append({"dataset": dataset, "seed": seed, "method": method_name, **rank})

                driver = driver_score_table(tagged, truth, variables=list(VARIABLES))
                driver_frames.append(driver.assign(dataset=dataset, seed=seed, method=method_name))

                runtime_rows.append({"dataset": dataset, "seed": seed, "method": method_name, "seconds": elapsed})
                print(f"finished dataset={dataset} seed={seed:02d} method={method_name} seconds={elapsed:.1f}", flush=True)

    metrics_by_run = pd.DataFrame(metrics_rows)
    ranking_by_run = pd.DataFrame(ranking_rows)
    driver_rankings = pd.concat(driver_frames, ignore_index=True) if driver_frames else pd.DataFrame()
    edge_results = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    ground_truth = pd.concat(truth_frames, ignore_index=True) if truth_frames else pd.DataFrame()
    summary = summarize_metrics(metrics_by_run)
    ranking_summary = summarize_ranking_metrics(ranking_by_run)
    stability = edge_stability(edge_results, seeds=args.seeds)
    runtime_log = pd.DataFrame(runtime_rows)
    method_params = method_parameters_frame(method_keys, args)
    method_params.loc[len(method_params)] = {
        "method": "GLOBAL",
        "parameter": "total_elapsed_seconds",
        "value": round(time.time() - started, 3),
    }

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary_binary_metrics", index=False)
        metrics_by_run.to_excel(writer, sheet_name="metrics_by_run", index=False)
        ranking_summary.to_excel(writer, sheet_name="summary_ranking_metrics", index=False)
        ranking_by_run.to_excel(writer, sheet_name="ranking_metrics_by_run", index=False)
        driver_rankings.to_excel(writer, sheet_name="driver_rankings", index=False)
        edge_results.to_excel(writer, sheet_name="edge_results", index=False)
        stability.to_excel(writer, sheet_name="edge_stability", index=False)
        ground_truth.to_excel(writer, sheet_name="ground_truth", index=False)
        method_params.to_excel(writer, sheet_name="method_parameters", index=False)
        runtime_log.to_excel(writer, sheet_name="runtime_log", index=False)

    return {
        "summary_binary_metrics": summary,
        "metrics_by_run": metrics_by_run,
        "summary_ranking_metrics": ranking_summary,
        "ranking_metrics_by_run": ranking_by_run,
        "driver_rankings": driver_rankings,
        "edge_results": edge_results,
        "edge_stability": stability,
        "ground_truth": ground_truth,
        "method_parameters": method_params,
        "runtime_log": runtime_log,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=DEFAULT_T)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_MODULES), default=list(SYNTHETIC_METHODS))
    parser.add_argument("--out", default=str(SYNTHETIC_RESULTS_XLSX))
    parser.add_argument("--data-dir", default=str(FINAL_SYNTHETIC_DATA_DIR))
    parser.add_argument("--log-dir", default=str(RUN_LOG_DIR))
    parser.add_argument("--cmiknn-sig-samples", type=int, default=CMIKNN_SIG_SAMPLES)
    parser.add_argument("--cmiknn-sig-blocklength", type=int, default=CMIKNN_SIG_BLOCKLENGTH)
    parser.add_argument("--cmiknn-workers", type=int, default=CMIKNN_WORKERS)
    parser.add_argument("--te-n-perm", type=int, default=SYNTHETIC_TE_N_PERM)
    parser.add_argument("--te-fdr", action="store_true", default=TE_FDR_CORRECTION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.dry_run:
        dry_run(parsed)
    else:
        run_synthetic_benchmark(parsed)
