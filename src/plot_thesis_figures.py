"""Generate publication-ready figures for the causal benchmark thesis.

The script reads completed result workbooks and only performs plotting. It does
not rerun VAR-GC, PCMCI, Transfer Entropy, or any data-generation process.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import FINAL_FIGURE_DIR, REAL_MARKET_RESULTS_XLSX, SYNTHETIC_RESULTS_XLSX

DEFAULT_SYNTHETIC = str(SYNTHETIC_RESULTS_XLSX)
DEFAULT_REAL = str(REAL_MARKET_RESULTS_XLSX)
DEFAULT_OUT_DIR = str(FINAL_FIGURE_DIR)

SCENARIOS = [
    "S0_linear_mixed_strength",
    "S1_indirect_conditional",
    "S2_hidden_confounder",
    "S3_nonlinear_nonmonotonic",
    "S4_feedback_mixed_lag",
]
SCENARIO_LABELS = {
    "S0_linear_mixed_strength": "S0",
    "S1_indirect_conditional": "S1",
    "S2_hidden_confounder": "S2",
    "S3_nonlinear_nonmonotonic": "S3",
    "S4_feedback_mixed_lag": "S4",
}
SCENARIO_LONG_LABELS = {
    "S0_linear_mixed_strength": "S0\nLinear mixed",
    "S1_indirect_conditional": "S1\nConditional",
    "S2_hidden_confounder": "S2\nHidden",
    "S3_nonlinear_nonmonotonic": "S3\nNonlinear",
    "S4_feedback_mixed_lag": "S4\nFeedback",
}
METHODS = ["VAR-GC", "PCMCI-ParCorr", "TE-IDTxl-MultivariateTE"]
METHOD_LABELS = {
    "VAR-GC": "VAR-GC",
    "PCMCI-ParCorr": "PCMCI-ParCorr",
    "TE-IDTxl-MultivariateTE": "TE",
}
METHOD_COLORS = {
    "VAR-GC": "#4C78A8",
    "PCMCI-ParCorr": "#F58518",
    "TE-IDTxl-MultivariateTE": "#54A24B",
}
REAL_LABELS = {
    "d_yield_1y": "1Y yield",
    "d_yield_10y": "10Y yield",
    "fx_log_return": "FX",
    "hs300_log_return": "CSI 300",
    "d_shibor_1w": "SHIBOR",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def read_first_available_sheet(workbook: Path, sheet_names: list[str]) -> pd.DataFrame:
    xl = pd.ExcelFile(workbook)
    available = set(xl.sheet_names)
    for sheet in sheet_names:
        if sheet in available:
            return pd.read_excel(workbook, sheet_name=sheet)
    raise ValueError(f"None of the requested sheets exist in {workbook}: {sheet_names}")


def clean_axis(ax: plt.Axes, ylim: tuple[float, float] | None = None) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)


def ordered_subset(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([SCENARIOS, METHODS], names=["dataset", "method"])
    cols = ["dataset", "method", f"{metric}_mean", f"{metric}_std"]
    out = frame[cols].set_index(["dataset", "method"]).reindex(index).reset_index()
    out[f"{metric}_std"] = pd.to_numeric(out[f"{metric}_std"], errors="coerce").fillna(0.0)
    out[f"{metric}_mean"] = pd.to_numeric(out[f"{metric}_mean"], errors="coerce")
    return out


def grouped_bar_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    data = ordered_subset(frame, metric)
    x = np.arange(len(SCENARIOS))
    width = 0.23
    offsets = np.linspace(-width, width, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        sub = data[data["method"] == method]
        ax.bar(
            x + offset,
            sub[f"{metric}_mean"].to_numpy(dtype=float),
            width=width,
            yerr=sub[f"{metric}_std"].to_numpy(dtype=float),
            capsize=2.0,
            linewidth=0.5,
            edgecolor="white",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    clean_axis(ax, ylim=ylim)


def plot_binary_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 score"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (metric, title) in zip(axes, specs):
        grouped_bar_panel(ax, summary, metric, title, "Mean over seeds", (0.0, 1.08))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Binary Edge Recovery", y=1.16, fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure_1_synthetic_binary_edge_recovery")


def plot_ranking_metrics(ranking: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("edge_auroc_p_score", "Edge AUROC\np-score"),
        ("edge_auprc_p_score", "Edge AUPRC\np-score"),
        ("edge_auroc_effect_score", "Edge AUROC\neffect-score"),
        ("edge_auprc_effect_score", "Edge AUPRC\neffect-score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=True)
    for ax, (metric, title) in zip(axes.ravel(), specs):
        grouped_bar_panel(ax, ranking, metric, title, "Mean over seeds", (0.0, 1.08))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Threshold-Free Edge Ranking", y=1.08, fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure_2_synthetic_edge_ranking")


def plot_driver_recovery(ranking: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("spearman_driver_p_score", "Driver Spearman\np-score", (-0.15, 1.08)),
        ("spearman_driver_effect_score", "Driver Spearman\neffect-score", (-0.15, 1.08)),
        ("top3_driver_recall_p_score", "Top-3 Driver Recall\np-score", (0.0, 1.08)),
        ("top3_driver_recall_effect_score", "Top-3 Driver Recall\neffect-score", (0.0, 1.08)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for ax, (metric, title, ylim) in zip(axes.ravel(), specs):
        grouped_bar_panel(ax, ranking, metric, title, "Mean over seeds", ylim)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Driver-Ranking Recovery", y=1.08, fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure_3_synthetic_driver_ranking_recovery")


def plot_driver_score_heatmap(driver: pd.DataFrame, out_dir: Path) -> None:
    frame = driver.copy()
    frame["method"] = pd.Categorical(frame["method"], METHODS, ordered=True)
    frame["dataset"] = pd.Categorical(frame["dataset"], SCENARIOS, ordered=True)
    estimated = (
        frame.groupby(["dataset", "method", "variable"], observed=False)["estimated_driver_effect_score"]
        .mean()
        .reset_index()
    )
    truth = (
        frame.groupby(["dataset", "variable"], observed=False)["true_driver_score"]
        .mean()
        .reset_index()
    )
    variables = sorted(frame["variable"].dropna().unique())

    truth_matrix = (
        truth.pivot(index="dataset", columns="variable", values="true_driver_score")
        .reindex(index=SCENARIOS, columns=variables)
        .fillna(0.0)
    )
    panels: list[tuple[str, pd.DataFrame, bool]] = [("Ground Truth", truth_matrix, True)]
    for method in METHODS:
        method_matrix = (
            estimated[estimated["method"] == method]
            .pivot(index="dataset", columns="variable", values="estimated_driver_effect_score")
            .reindex(index=SCENARIOS, columns=variables)
            .fillna(0.0)
        )
        panels.append((METHOD_LABELS[method], method_matrix, False))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    im = None
    for panel_index, (ax, (title, matrix, annotate_truth)) in enumerate(zip(axes.ravel(), panels)):
        raw_values = matrix.to_numpy(dtype=float)
        panel_max = np.nanmax(raw_values)
        if np.isfinite(panel_max) and panel_max > 0:
            normalized = raw_values / panel_max
        else:
            normalized = np.zeros_like(raw_values)
        im = ax.imshow(normalized, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(variables)))
        ax.set_xticklabels(variables)
        ax.set_yticks(np.arange(len(SCENARIOS)))
        if panel_index % 2 == 0:
            ax.set_yticklabels([SCENARIO_LONG_LABELS[s].replace("\n", " ") for s in SCENARIOS])
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if annotate_truth:
            for row in range(raw_values.shape[0]):
                for col in range(raw_values.shape[1]):
                    color = "white" if normalized[row, col] >= 0.62 else "black"
                    ax.text(
                        col,
                        row,
                        f"{raw_values[row, col]:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=6.5,
                    )

    colorbar_ax = fig.add_axes([0.89, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=colorbar_ax, label="Within-panel normalized driver score")
    fig.suptitle("True and Estimated Driver Strength by Method", y=0.97, fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.10, top=0.89, hspace=0.32, wspace=0.18)
    save_figure(fig, out_dir, "Figure_4_true_and_estimated_driver_strength")


def plot_real_edge_counts(significant: pd.DataFrame, out_dir: Path) -> None:
    counts = significant.groupby("method").size().reindex(METHODS, fill_value=0)
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    ax.bar(
        [METHOD_LABELS[m] for m in METHODS],
        counts.to_numpy(dtype=float),
        color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Number of significant edges")
    ax.set_title("Real Market Significant Edges")
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure_5_real_market_significant_edge_counts")


def plot_real_net_degree(network: pd.DataFrame, out_dir: Path) -> None:
    variables = list(dict.fromkeys(network["variable"].astype(str)))
    variables = [v for v in REAL_LABELS if v in variables] + [v for v in variables if v not in REAL_LABELS]
    x = np.arange(len(variables))
    width = 0.23
    offsets = np.linspace(-width, width, len(METHODS))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for offset, method in zip(offsets, METHODS):
        sub = (
            network[network["method"] == method]
            .set_index("variable")
            .reindex(variables)
            .fillna({"net_out_degree": 0})
        )
        ax.bar(
            x + offset,
            sub["net_out_degree"].to_numpy(dtype=float),
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Out-degree minus in-degree")
    ax.set_title("Real Market Net Causal Degree")
    ax.set_xticks(x)
    ax.set_xticklabels([REAL_LABELS.get(v, v) for v in variables], rotation=20, ha="right")
    clean_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure_6_real_market_net_degree")


def plot_real_consensus_matrix(consensus: pd.DataFrame, all_edges: pd.DataFrame, out_dir: Path) -> None:
    variables = list(dict.fromkeys(pd.concat([all_edges["source"], all_edges["target"]]).astype(str)))
    variables = [v for v in REAL_LABELS if v in variables] + [v for v in variables if v not in REAL_LABELS]
    matrix = pd.DataFrame(0.0, index=variables, columns=variables)
    for row in consensus.itertuples(index=False):
        matrix.loc[str(row.source), str(row.target)] = float(row.method_count)
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=max(3, matrix.to_numpy().max()))
    ax.set_xticks(np.arange(len(variables)))
    ax.set_yticks(np.arange(len(variables)))
    ax.set_xticklabels([REAL_LABELS.get(v, v) for v in variables], rotation=35, ha="right")
    ax.set_yticklabels([REAL_LABELS.get(v, v) for v in variables])
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title("Real Market Consensus Edges")
    for i, source in enumerate(variables):
        for j, target in enumerate(variables):
            if source == target:
                text = ""
            else:
                value = int(matrix.iloc[i, j])
                text = str(value) if value else ""
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Number of methods")
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure_7_real_market_consensus_matrix")


def plot_synthetic_figures(workbook: Path, out_dir: Path) -> None:
    summary = read_first_available_sheet(workbook, ["summary_binary_metrics", "summary"])
    ranking = read_first_available_sheet(workbook, ["summary_ranking_metrics", "ranking_summary"])
    driver = read_first_available_sheet(workbook, ["driver_rankings"])
    plot_binary_metrics(summary, out_dir)
    plot_ranking_metrics(ranking, out_dir)
    plot_driver_recovery(ranking, out_dir)
    plot_driver_score_heatmap(driver, out_dir)


def plot_real_figures(workbook: Path, out_dir: Path) -> None:
    if not workbook.exists():
        return
    significant = pd.read_excel(workbook, sheet_name="significant_edges")
    consensus = pd.read_excel(workbook, sheet_name="consensus_edges")
    network = pd.read_excel(workbook, sheet_name="network_summary")
    all_edges = pd.read_excel(workbook, sheet_name="all_edges")
    plot_real_edge_counts(significant, out_dir)
    plot_real_net_degree(network, out_dir)
    plot_real_consensus_matrix(consensus, all_edges, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=DEFAULT_SYNTHETIC)
    parser.add_argument("--real", default=DEFAULT_REAL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-real", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    synthetic = Path(args.synthetic)
    real = Path(args.real)
    out_dir = Path(args.out_dir)
    if not synthetic.exists():
        raise FileNotFoundError(f"Synthetic workbook not found: {synthetic}")
    plot_synthetic_figures(synthetic, out_dir)
    if not args.skip_real:
        plot_real_figures(real, out_dir)
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
