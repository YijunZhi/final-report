# Final Causal Benchmark Workflow

The final thesis workflow is organised as one Python file per task:

- `src/config.py`: shared parameters and paths.
- `src/synthetic_data.py`: synthetic S0-S4 data generation and ground truth.
- `src/methods/`: formal causal discovery methods.
- `src/metrics.py`: binary edge metrics, ranking metrics, driver recovery, and stability.
- `src/run_synthetic_benchmark.py`: synthetic benchmark runner.
- `src/run_real_market.py`: real market runner.
- `src/plot_thesis_figures.py`: publication-ready figures.
- `src/run_all.py`: end-to-end final workflow.

Run the complete final workflow with:

```powershell
conda activate mlbd
python src\run_all.py
```

Final outputs:

- `results/final/synthetic_results.xlsx`
- `results/final/real_market_results.xlsx`
- `results/final/figures/`

The final workflow uses:

- synthetic benchmark: `TE n_perm=25`
- real market benchmark: `TE n_perm=100`
- methods: `VAR-GC`, `PCMCI-ParCorr`, and `TE-IDTxl-MultivariateTE`
- `PCMCI-CMIknn` remains available in `src/methods/`, but is not run by default.

## Formal Synthetic Causal Benchmark

This folder is organised for four formal methods only:

- `VAR-GC`: `statsmodels` VAR-Granger causality.
- `PCMCI-ParCorr`: `tigramite` PCMCI with ParCorr.
- `PCMCI-CMIknn`: `tigramite` PCMCI with CMIknn.
- `TE-IDTxl-MultivariateTE`: IDTxl MultivariateTE with JidtKraskovCMI.

No approximate `TE-Discrete` result is used in the formal runner.

## Default Benchmark

```powershell
conda activate mlbd
python src\run_synthetic_benchmark.py
```

Default parameters:

- `T=500`
- `seeds=10`, meaning seed values `0..9`
- `datasets=S0_linear_mixed_strength,S1_indirect_conditional,S2_hidden_confounder,S3_nonlinear_nonmonotonic,S4_feedback_mixed_lag`
- `variables=X1,X2,X3,X4,X5,X6`
- `max_lag=5`
- `alpha=0.05`
- `burn_in=500`
- Gaussian innovation noise

Synthetic datasets:

- `S0_linear_mixed_strength`: signed linear edges with strong, medium, and weak effects across lags 1-5.
- `S1_indirect_conditional`: chains, an observed fork, and a collider; indirect shortcut edges are deliberately absent.
- `S2_hidden_confounder`: hidden `U` drives `X1` and `X2` at different lags; `U` is stored only in metadata.
- `S3_nonlinear_nonmonotonic`: even, two-sided threshold, saturating, and weak linear effects.
- `S4_feedback_mixed_lag`: two asymmetric feedback pairs plus a long-lag transmission edge.

Output:

- `results/final/synthetic_results.xlsx`
- `data/final_synthetic/`

Main Excel sheets:

- `summary_binary_metrics`: binary edge recovery metrics based on significant edges, including Precision, Recall, F1, SHD, exact-lag accuracy, and strong/medium/weak-edge recall.
- `metrics_by_run`: seed-level binary edge recovery metrics.
- `summary_ranking_metrics`: threshold-free Edge AUROC/AUPRC and driver-ranking recovery summarized by dataset and method.
- `ranking_metrics_by_run`: seed-level Edge AUROC/AUPRC and driver-ranking recovery.
- `driver_rankings`: variable-level true and estimated driver scores.
- `edge_results`: all ordered-pair method outputs.
- `ground_truth`: true synthetic edges, including structural `effect_size`.

## Dry Run

Use this to validate imports and print method parameters without running the
benchmark:

```powershell
conda activate mlbd
python src\run_synthetic_benchmark.py --dry-run
```

## Method Parameters

All runtime parameters are written to the `method_parameters` sheet of the
Excel output.  The most important defaults are:

- VAR-GC: BIC lag selection up to `max_lag=5`; FDR-BH over ordered pairs.
- PCMCI-ParCorr: `tau_min=1`, `tau_max=5`, analytic ParCorr, FDR-BH corrected p-values.
- PCMCI-CMIknn: `knn=0.2`, `shuffle_neighbors=5`, `sig_samples=100`,
  `sig_blocklength=5`, `transform='ranks'`, `workers=-1`.
- TE-IDTxl: `JidtKraskovCMI`, `n_perm=25`, `kraskov_k=4`,
  `permute_in_time=True`, network-level `fdr_correction=False` by default.

`PCMCI-CMIknn` and `TE-IDTxl` are much slower than VAR-GC and PCMCI-ParCorr.

## Ranking Evaluation

The benchmark also reports two threshold-free method-comparison diagnostics.

- Edge AUROC/AUPRC ranks the 30 ordered source-target pairs, `Xi -> Xj`,
  by continuous method scores rather than thresholding only at `p <= 0.05`.
- Driver-ranking recovery sums outgoing edge scores for each source variable
  and compares estimated driver rankings with the true driver strengths.

Binary evaluation additionally reports exact-lag accuracy among correctly
detected directed edges and separate recall for strong (`|effect| >= 0.35`),
medium (`0.22 <= |effect| < 0.35`), and weak (`|effect| < 0.22`) true edges.

The true driver strengths are based on the synthetic structural coefficients
stored as `effect_size` in `ground_truth`.
