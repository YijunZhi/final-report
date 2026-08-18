"""Run the final thesis workflow end to end.

Default steps:
1. synthetic benchmark: VAR-GC, PCMCI-ParCorr, and TE
2. real market benchmark: VAR-GC, PCMCI-ParCorr, and TE
3. publication-ready figures
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from config import (
    CMIKNN_SIG_BLOCKLENGTH,
    CMIKNN_SIG_SAMPLES,
    CMIKNN_WORKERS,
    DEFAULT_ALPHA,
    DEFAULT_MAX_LAG,
    DEFAULT_SEEDS,
    DEFAULT_T,
    FINAL_FIGURE_DIR,
    FINAL_SYNTHETIC_DATA_DIR,
    REAL_MARKET_INPUT_CSV,
    REAL_MARKET_METHODS,
    REAL_MARKET_RESULTS_XLSX,
    REAL_MARKET_TE_N_PERM,
    RUN_LOG_DIR,
    SYNTHETIC_METHODS,
    SYNTHETIC_RESULTS_XLSX,
    SYNTHETIC_TE_N_PERM,
    TE_FDR_CORRECTION,
)
from plot_thesis_figures import configure_matplotlib, plot_real_figures, plot_synthetic_figures
from run_real_market import run_real_market
from run_synthetic_benchmark import run_synthetic_benchmark


def synthetic_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        t=args.t,
        seeds=args.seeds,
        max_lag=args.max_lag,
        alpha=args.alpha,
        methods=list(args.synthetic_methods),
        out=str(args.synthetic_out),
        data_dir=str(args.synthetic_data_dir),
        log_dir=str(args.log_dir),
        cmiknn_sig_samples=CMIKNN_SIG_SAMPLES,
        cmiknn_sig_blocklength=CMIKNN_SIG_BLOCKLENGTH,
        cmiknn_workers=CMIKNN_WORKERS,
        te_n_perm=args.synthetic_te_n_perm,
        te_fdr=TE_FDR_CORRECTION,
        dry_run=False,
    )


def real_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        input=str(args.real_input),
        out=str(args.real_out),
        max_lag=args.max_lag,
        alpha=args.alpha,
        methods=list(args.real_methods),
        te_n_perm=args.real_te_n_perm,
        te_fdr=TE_FDR_CORRECTION,
        dry_run=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=DEFAULT_T)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--synthetic-methods", nargs="+", default=list(SYNTHETIC_METHODS))
    parser.add_argument("--real-methods", nargs="+", default=list(REAL_MARKET_METHODS))
    parser.add_argument("--synthetic-te-n-perm", type=int, default=SYNTHETIC_TE_N_PERM)
    parser.add_argument("--real-te-n-perm", type=int, default=REAL_MARKET_TE_N_PERM)
    parser.add_argument("--synthetic-out", type=Path, default=SYNTHETIC_RESULTS_XLSX)
    parser.add_argument("--real-out", type=Path, default=REAL_MARKET_RESULTS_XLSX)
    parser.add_argument("--real-input", type=Path, default=REAL_MARKET_INPUT_CSV)
    parser.add_argument("--synthetic-data-dir", type=Path, default=FINAL_SYNTHETIC_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FINAL_FIGURE_DIR)
    parser.add_argument("--log-dir", type=Path, default=RUN_LOG_DIR)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_synthetic:
        print("=== Running synthetic benchmark ===", flush=True)
        run_synthetic_benchmark(synthetic_args(args))
        print(f"synthetic workbook: {args.synthetic_out}", flush=True)

    if not args.skip_real:
        print("=== Running real market benchmark ===", flush=True)
        run_real_market(real_args(args))
        print(f"real market workbook: {args.real_out}", flush=True)

    if not args.skip_figures:
        print("=== Generating thesis figures ===", flush=True)
        configure_matplotlib()
        plot_synthetic_figures(Path(args.synthetic_out), Path(args.figure_dir))
        if not args.skip_real:
            plot_real_figures(Path(args.real_out), Path(args.figure_dir))
        else:
            plot_real_figures(Path(args.real_out), Path(args.figure_dir))
        print(f"figures: {args.figure_dir}", flush=True)

    print("=== Final workflow complete ===", flush=True)


if __name__ == "__main__":
    main()
