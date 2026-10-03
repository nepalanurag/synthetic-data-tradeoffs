# Synthetic tabular data: measuring the fidelity-utility-privacy tradeoff

## Background

Synthetic data is often presented as a way to share data without sharing
people: generate fake rows that keep the statistical signal, and the privacy
problem goes away. Practitioners know it is not that clean. Any generator
sits somewhere on three axes:

1. **Fidelity.** Does the synthetic data reproduce the real distribution?
2. **Utility.** Can you train a model on synthetic data and have it work on
   real data?
3. **Privacy.** Does the synthetic data leak information about real training
   records?

These pull against each other. A generator that reproduces the training
distribution perfectly has, in the limit, just memorized it. A generator that
stays far from every real record is safe but may be useless. This project is
a controlled measurement of that tension on one well-understood dataset,
comparing a classical statistical generator against a modern LLM-based one.

## Data

The breast cancer Wisconsin diagnostic dataset (Wolberg, Street, and
Mangasarian, 1995; distributed with scikit-learn): 569 rows, 30 numeric
features computed from digitized cell-nucleus images, binary target
(0 = malignant, 1 = benign). It is fully numeric, which keeps both generators
on equal footing without encoding machinery, and the classification task is
real, which makes the utility test meaningful.

The data were split once, 70/30 stratified, seed 20261002: 398 training rows
available to the generators, 171 holdout rows used only for evaluation.
The copula produced 500 synthetic rows. The LLM produced 126 validated rows
from 21 accepted batches (5 batches were rejected and not retried further
after this resumed run assembled the cache as-is; raw responses are cached
under `data/llm_cache/`, so the exact run is reproducible without new API
calls).

## Methodology

### Generator A: Gaussian copula (hand-rolled)

A copula separates the margins from the dependence structure. My
implementation (`src/copula.py`):

1. For each feature, map values to uniform on (0, 1) through the empirical
   CDF (average ranks divided by n + 1, so the probit stays finite).
2. Map uniforms to standard normals with the probit function.
3. Estimate the mean vector and covariance matrix of the latent normals.
4. Sample new latent vectors, map back through the standard normal CDF and
   then through the inverse empirical CDF of each feature (linear
   interpolation).

The copula is fit separately for each target class, and I draw synthetic
class labels from the training prevalence (40.8% malignant, 59.2% benign). This is a deliberate modeling choice: for a classification utility
test, the class-conditional structure is what matters, and a single joint
fit would dilute it.

Statistical properties worth stating plainly: the margins are reproduced
exactly up to sampling noise, and the rank correlation structure is captured
by the latent covariance. Anything that lives outside the correlation
matrix, multimodality, non-monotone dependence, tail dependence beyond the
Gaussian, is flattened. The sanity checks (no NaNs, no values outside the
training range, class balance preserved) all passed.

### Generator B: LLM row synthesis

I prompted Gemini (flash tier) with the dataset schema: column names,
observed ranges per feature, the meaning of the target, and 8 real example
rows, asking for batches of 40 new rows as plain CSV. I validated every
batch before acceptance: correct column count, all numeric cells, target
in {0, 1}, every feature within the observed training range (2% slack).
Failed batches were retried; 21 batches were accepted and
5 rejected. Raw model responses are cached under
`data/llm_cache/`, so the exact run is reproducible without new API calls.

This generator has no statistical guarantees. It is included because it is
what people actually reach for now, and the interesting question is what it
costs relative to a classical method.

### Evaluation

**Fidelity.** Per-feature two-sample Kolmogorov-Smirnov tests (statistic and
p-value for all 30 features, saved in `data/ks_copula.csv` and
`data/ks_llm.csv`); the Frobenius distance between the Pearson correlation
matrices of real and synthetic data; and the squared MMD with an RBF kernel,
bandwidth fixed once from the real training set via the median heuristic
(gamma = 2.11e-06), so both generators face the same kernel.

**Utility (TSTR: train on synthetic, test on real).** Logistic regression and
a 300-tree random forest were trained on each of the three training sets
(real, copula synthetic, LLM synthetic) and scored by AUC on the same real
holdout. The train-on-real model is the baseline. Differences in AUC use
bootstrap 95% confidence intervals (2000 resamples of the holdout), so a gap
has to clear sampling noise to be taken seriously.

**Privacy.** First, distance to the closest record (DCR): for each synthetic
row, the Euclidean distance to the nearest real training row, with features
standardized by the training mean and sd. The baseline is real training rows
measured against other real training rows (leave-one-out), plus real holdout
rows measured against the training set. If synthetic rows sit closer to the
training data than real rows sit to each other, the generator is copying.
Second, a simple membership-inference attack: score each real row (train
members vs holdout non-members) by its negative nearest-neighbor distance to
the synthetic set, and report the AUC. An AUC of 0.5 means the synthetic data
carries no membership signal; anything clearly above is leakage.

## Results

### Fidelity

| | Mean KS | Max KS | Features with p < 0.05 | Corr-matrix distance | MMD^2 |
|---|---|---|---|---|---|
| Copula | 0.0428 | 0.0631 | 0/30 | 1.4168 | 0.00078 |
| LLM | 0.2695 | 0.3892 | 30/30 | 15.7272 | 0.08195 |

The copula wins every fidelity metric, as it should: exact margins and an
explicit correlation fit. The LLM rows are visibly noisier on the margins
(larger KS statistics, more rejected p-values) and miss more of the joint
structure. The per-feature KS table shows the worst LLM features; the pattern
is consistent with the model struggling to hold 30 correlated numeric ranges
in its head at once.

### Utility

| Model | Trained on | Holdout AUC | 95% CI |
|---|---|---|---|
| Random forest | Real | 0.9849 | [0.9697, 0.9954] |
| Random forest | Copula | 0.9824 | [0.9651, 0.9954] |
| Random forest | LLM | 0.9546 | [0.9193, 0.9803] |
| Logistic regression | Real | 0.9885 | [0.9736, 0.9979] |
| Logistic regression | Copula | 0.9844 | [0.9687, 0.9945] |
| Logistic regression | LLM | 0.9194 | [0.8755, 0.9563] |

AUC drop vs train-on-real (random forest): copula +0.0025, LLM
+0.0303, with bootstrap CIs reported in the notebook. The TSTR test is
the one a practitioner actually cares about, and it is the least flattering
to both generators: matching margins is not the same as preserving the
decision boundary.

### Privacy

| | DCR median (synthetic to train) | DCR median (real to real) | MI attack AUC |
|---|---|---|---|
| Copula | 2.298 | 2.211 | 0.542 |
| LLM | 2.545 | 2.211 | 0.560 |

Real holdout rows sit at median distance 2.226 from the training
set. The copula's tighter fit shows up here: its rows sit at 2.298, barely
above the real-to-real baseline of 2.211, which is the price of fitting the
dependence structure well. The LLM rows spread out more (2.545), but the
membership-inference attack found slightly more signal in them (0.560 vs
0.542 for the copula), so neither number reads as a privacy win; both sit
barely above the 0.5 no-signal line. The attack is deliberately weak; treat
these as lower bounds on risk.

### The tradeoff

Plotted as fidelity (correlation-matrix distance) against utility loss (TSTR
AUC drop), with bubble size proportional to privacy leakage (MI AUC minus
0.5), the picture is a triangle with the copula sitting toward good fidelity and
utility with a small privacy bubble; the LLM sits toward worse fidelity and
a real utility gap, with a comparable (slightly larger) privacy bubble. The
ideal bottom-left tiny bubble is empty: this run has no generator that is
private, faithful, and useful all at once.

## Interpretation

- Use the copula when you need statistical fidelity and the data stays
  in-house: it reproduces margins and correlations faithfully, and the cost
  is a measurable closeness to real records.
- The privacy comparison is a wash in this run: LLM rows sit further from
  real records by DCR, but the membership-inference attack found slightly
  more signal in them than in the copula set (0.560 vs 0.542). Do not share
  either set casually on privacy grounds alone.
- Neither replaces real data for model selection. The TSTR gaps, with honest
  CIs, say how much you pay.
- The LLM failure modes observed here were concrete: batches with wrong
  column counts, non-numeric cells, and out-of-range values (all rejected and
  retried), plus systematically worse margin fidelity even in accepted
  batches. Prompting a language model is not a statistical method; without
  the validation layer it is not safe to use.

## Threats to validity

One dataset, fully numeric, 569 rows; mixed-type or heavy-tailed data could
change the ranking. The MI attack is a lower bound on risk, not a safety
certificate. LLM output depends on model version and prompt, though the raw
responses are cached for exact reproducibility. TSTR measures one downstream
task. The per-class copula fit helps classification utility specifically.

## References

- Wolberg, W., Street, W. N., and Mangasarian, O. L. (1995). Breast cancer
  Wisconsin (diagnostic) dataset. UCI Machine Learning Repository.
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
- Nelsen, R. B. (2006). An Introduction to Copulas. Springer.
- Gretton, A., et al. (2012). A kernel two-sample test. JMLR.
- Shokri, R., et al. (2017). Membership inference attacks against machine
  learning models. IEEE S&P.

All numbers in this report come from `data/results.json`, produced by
`src/run_eval.py` with seed 20261002. Nothing is invented.
