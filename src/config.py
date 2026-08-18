"""Central configuration for the final causal benchmark workflow."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_T = 500
DEFAULT_SEEDS = 10
DEFAULT_MAX_LAG = 5
DEFAULT_ALPHA = 0.05

SYNTHETIC_METHODS = ("vargc", "pcmci_parcorr", "te")
REAL_MARKET_METHODS = ("vargc", "pcmci_parcorr", "te")

SYNTHETIC_TE_N_PERM = 25
REAL_MARKET_TE_N_PERM = 100
TE_FDR_CORRECTION = False

CMIKNN_SIG_SAMPLES = 100
CMIKNN_SIG_BLOCKLENGTH = 5
CMIKNN_WORKERS = -1

FINAL_RESULTS_DIR = PROJECT_ROOT / "results" / "final"
FINAL_SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "final_synthetic"
FINAL_FIGURE_DIR = FINAL_RESULTS_DIR / "figures"
RUN_LOG_DIR = PROJECT_ROOT / "results" / "run_logs"

SYNTHETIC_RESULTS_XLSX = FINAL_RESULTS_DIR / "synthetic_results.xlsx"
REAL_MARKET_INPUT_CSV = PROJECT_ROOT / "data" / "real_market_2015_2025" / "real_market_stationary_zscore_2015_2025.csv"
REAL_MARKET_RESULTS_XLSX = FINAL_RESULTS_DIR / "real_market_results.xlsx"

METHOD_MODULES = {
    "vargc": "methods.vargc",
    "pcmci_parcorr": "methods.pcmci_parcorr",
    "pcmci_cmiknn": "methods.pcmci_cmiknn",
    "te": "methods.te_idtxl",
}
