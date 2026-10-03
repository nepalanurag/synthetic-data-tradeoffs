# dashboard-data: synthetic-data-tradeoffs

Key results from `notebooks/analysis.ipynb` ("Fidelity, utility, privacy:
what does synthetic tabular data actually cost?"), exported for the
interactive dashboard. All numbers come from the notebook's executed outputs
and the repo's `data/results.json`. Two generators (hand-built Gaussian
copula, Gemini row synthesis) on the breast cancer Wisconsin dataset; no
generator wins on all three axes.

## Files

- **datasets.json** — the real split and both synthetic sets.
  Fields: `n_features`, `target`, `seed`,
  `sets[]`: `name`, `rows`, `target_mean`, `nans`.
- **fidelity.json** — does the synthetic data look real?
  Fields: `n_features`,
  `ks[]`: `generator`, `mean_ks`, `max_ks`, `features_p_lt_0.05`,
  `worst_5_features_llm[]`: `feature`, `ks_stat`, `ks_pvalue`,
  `real_mean`, `syn_mean`,
  `correlation_matrix_distance`: `copula`, `llm`,
  `mmd_squared`: `gamma`, `copula`, `llm`.
- **utility.json** — train on synthetic, test on real (TSTR).
  Fields: `holdout_auc[]`: `model` ("logreg"/"rf"), `trained_on`, `auc`;
  `auc_diff_vs_real[]`: `model`, `trained_on`, `auc_diff`,
  `ci95` ([low, high], bootstrap 95%).
- **privacy.json** — how close is synthetic to real?
  Fields: `dcr[]`: `set`, `mean`, `median`, `p5` (distance to closest
  record); `membership_inference_auc`: `copula`, `llm`, `no_signal` (0.5).
- **tradeoff_summary.json** — one row per generator for the tradeoff plot.
  Fields: `generators[]`: `generator`, `corr_dist` (fidelity, x),
  `auc_drop_rf` (utility loss, y), `mi_auc` (privacy leakage, bubble size).
- **llm_generation.json** — the LLM synthesis run log.
  Fields: `batches_accepted`, `batches_rejected`,
  `rejections[]`: `batch`, `reason`, `class_balance`, `nans`.
