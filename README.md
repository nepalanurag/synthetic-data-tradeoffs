# Synthetic data tradeoffs: fidelity, utility, privacy

Two ways to generate synthetic tabular data, scored on the three axes that
actually matter. No generator wins on all three; this project measures the
tradeoffs instead of picking a favorite.

## The question

Synthetic data is sold as a fix for data sharing: same signal, no privacy
risk. The honest version is a three-way tradeoff. A generator can match the
real distribution closely (fidelity), keep downstream models working (utility),
or stay far from real records (privacy). Pushing one usually costs the others.
This project quantifies that on a real dataset with real computed numbers.

## Data

The breast cancer Wisconsin dataset (scikit-learn, 569 rows, 30 numeric
features describing cell nuclei, target 0 = malignant / 1 = benign). Why this
one: fully numeric features so both a copula and an LLM can handle it without
extra encoding; a real binary classification task that makes the utility test
meaningful; public and standard so anyone can rerun this.

Split: fixed 70/30 stratified split, seed 20261002. 398 train rows the
generators may see, 171 holdout rows they may not. The copula produced 500
synthetic rows; the LLM produced 126 validated rows (21 batches accepted,
5 rejected; some model responses were partial or unparseable).

## Methods

**Generator A: Gaussian copula, implemented by hand** (`src/copula.py`).
I rank-transform each feature to uniform via the empirical CDF, map to
standard normals with the probit, fit mean and covariance, sample, and map
back through the inverse empirical CDF. I fit it separately per class and
sample class labels from the training prevalence. This reproduces every
margin exactly and the rank correlations, but nothing beyond the correlation
matrix. It cannot represent multimodality or non-monotone dependence.

**Generator B: LLM row synthesis** (`src/llm_synth.py`). I prompted Gemini
(flash tier) with the schema (column names, observed ranges, target meaning)
plus 8 real example rows, and asked for batches of 40 CSV rows. I validated
every batch (column count, numeric cells, 0/1 target, values within observed
ranges) and retried the bad ones: 21 accepted,
5 rejected. Raw API responses are cached in `data/llm_cache/`
so the run is reproducible.

**Evaluation** (`src/metrics.py`):
- Fidelity: per-feature two-sample KS tests; correlation-matrix distance
  (Frobenius norm); MMD with RBF kernel, bandwidth from the median heuristic.
- Utility (TSTR): logistic regression and random forest trained on each
  synthetic set, AUC on the real holdout; train-on-real baseline; AUC
  differences with bootstrap 95% CIs (2000 resamples).
- Privacy: distance-to-closest-record (DCR) distributions, synthetic-to-train
  vs real-to-real baseline; a threshold membership-inference attack on
  nearest-neighbor distance to the synthetic set (0.5 = no signal).

## Key results

Fidelity (lower is better):
- Per-feature KS, mean: copula 0.0428, LLM 0.2695
  (max: 0.0631 vs 0.3892; features with p < 0.05:
  0/30 vs 30/30).
- Correlation-matrix distance: copula 1.4168, LLM 15.7272.
- MMD squared (RBF): copula 0.00078, LLM 0.08195.

Utility, TSTR AUC on the real holdout (95% bootstrap CI):
- Random forest: real 0.9849 [0.9697, 0.9954],
  copula 0.9824 [0.9651, 0.9954],
  LLM 0.9546 [0.9193, 0.9803].
- Logistic regression: real 0.9885, copula 0.9844,
  LLM 0.9194.
- AUC drop vs train-on-real (RF): copula +0.0025, LLM +0.0303.

Privacy:
- DCR median, synthetic to train: copula 2.298, LLM 2.545
  (real-to-real baseline 2.211, real holdout to train 2.226).
- Membership-inference attack AUC: copula 0.542, LLM 0.560
  (0.5 means no leakage signal).

What I found: the copula wins fidelity by construction. The LLM rows are
noisier on the margins and spread out more (higher DCR), but the privacy
picture is mixed: the simple membership-inference attack found slightly more
signal in the LLM set (0.560 vs 0.542), both barely above the 0.5 no-signal
line. On utility, the copula's AUC gap vs train-on-real is within noise
(-0.0025, 95% CI crossing zero); the LLM's gap is real (-0.0303, CI fully
below zero).

## Limitations and threats to validity

- One dataset, fully numeric, 569 rows. Mixed-type, high-dimensional, or
  heavy-tailed data could rank the generators differently.
- The membership-inference attack is deliberately simple. These numbers are a
  lower bound on privacy risk, not a safety certificate.
- LLM synthesis depends on the model version and prompt. Raw responses are
  cached in the repo so this exact run reproduces; a different model may differ.
- TSTR measures one downstream task. A different task could change the ranking.
- The copula is fit per class, which helps classification utility but is the
  wrong choice if the target itself needed joint modeling.

## How to run

Requires Python 3 with numpy, pandas, scikit-learn, scipy, matplotlib
(installed). The LLM step needs the stored Gemini credential and is already
cached, so a rerun uses the cache without new API calls.

```
cd synthetic-data-tradeoffs
python3 src/make_base_data.py   # train/holdout split + copula synthetic set
# LLM synthesis (needs the stored Gemini credential; raw responses cached):
python3 src/synthesize_llm.py   # writes data/X_llm.npy, data/y_llm.npy
python3 src/run_eval.py         # metrics, data/results.json, figures
python3 src/build_notebook.py   # assemble notebooks/analysis.ipynb
python3 src/execute_notebook.py notebooks/analysis.ipynb  # run it, outputs saved
python3 src/fill_docs.py        # fill computed numbers into README/REPORT
python3 src/build_dashboard.py  # write dashboard.html
```

## Repo structure

```
src/
  copula.py            hand-rolled Gaussian copula (per-class fit)
  llm_synth.py         Gemini row synthesis with validation and caching
  metrics.py           KS / correlation distance / MMD / TSTR / DCR / MI attack
  plotting.py          matplotlib figures
  run_eval.py          metrics, results.json, figures
  build_notebook.py    assembles notebooks/analysis.ipynb
  execute_notebook.py  runs the notebook and saves outputs
  build_dashboard.py   writes dashboard.html
notebooks/analysis.ipynb   executed analysis with narrative
data/
  train.csv, holdout.csv, copula_synthetic.csv, llm_synthetic.csv
  llm_cache/           raw Gemini responses (reproducibility)
  results.json         every computed number
  ks_copula.csv, ks_llm.csv, llm_stats.json
figures/               KS bars, DCR histograms, tradeoff, TSTR forest plot
dashboard.html         self-contained Plotly dashboard
REPORT.md              full write-up
```

Seed 20261002 everywhere. No numbers in this file are invented; they come
from `data/results.json`.
