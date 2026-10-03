"""Build notebooks/analysis.ipynb from text cells defined here (code cells are
plain strings; the notebook is executed by src/execute_notebook.py, which
populates outputs)."""

import nbformat as nbf

ROOT_CELLS = []


def md(text):
    ROOT_CELLS.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    ROOT_CELLS.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Fidelity, utility, privacy: what does synthetic tabular data actually cost?

This notebook compares two ways of generating synthetic rows for a tabular
medical dataset, and scores both on three axes that matter in practice:

1. **Fidelity**: does the synthetic data look like the real data?
2. **Utility**: can you train a model on synthetic data and have it work on real data (TSTR)?
3. **Privacy**: does the synthetic data leak the real training rows?

The honest answer is that no generator wins on all three at once. The point
of this study is to measure the tradeoffs instead of picking a favorite.
""")

code("""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261002
rng = np.random.default_rng(SEED)
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data")
""")

md("""
## Setup (the data)

The input files are all committed (`data/train.csv`, `data/holdout.csv`,
`data/copula_synthetic.csv`, `data/llm_synthetic.csv`, `data/llm_cache/`,
`data/results.json`), so this notebook reruns without new API calls.

### The dataset

I use the breast cancer Wisconsin dataset (scikit-learn, 569 rows, 30 numeric
features of cell nuclei, target 0 = malignant / 1 = benign). Why this one:

- fully numeric features, so both a copula and an LLM can handle it without
  extra encoding machinery;
- a real binary classification task with a known structure, which makes the
  TSTR utility test meaningful;
- public and standard, so anyone can rerun this.

The split is a fixed 70/30 stratified split with seed 20261002: 398 train
rows that the generators are allowed to see, 171 holdout rows that they are
not. Everything downstream is measured against these fixed sets.
""")

code("""
train = pd.read_csv(os.path.join(DATA, "train.csv"))
hold = pd.read_csv(os.path.join(DATA, "holdout.csv"))
cop = pd.read_csv(os.path.join(DATA, "copula_synthetic.csv"))
llm = pd.read_csv(os.path.join(DATA, "llm_synthetic.csv"))
feature_names = [c for c in train.columns if c != "target"]
for name, df in [("train", train), ("holdout", hold), ("copula", cop), ("llm", llm)]:
    print(f"{name:8s} rows={len(df):4d}  target mean={df['target'].mean():.3f}  NaNs={int(df.isna().sum().sum())}")
""")

md("""
## Method""")

md("""
### Generator A: a Gaussian copula, implemented by hand

The idea: keep every univariate margin exactly as observed (rank transform),
and model only the dependence structure with a multivariate normal in
latent space.

1. For each feature, map values to uniform on (0,1) through the empirical
   CDF (ranks divided by n+1).
2. Map uniforms to standard normals with the probit function.
3. Fit a mean vector and covariance matrix to the latent normals.
4. To sample: draw latent normals, map back through the normal CDF and then
   through the inverse empirical CDF of each feature.

I fit one copula per class and sample class labels from the training
prevalence. This is a deliberate choice: the TSTR test needs the
class-conditional structure to be right, and a single joint copula would
dilute it. The cost is that we are not modeling the full joint distribution
as one block, which is fine here because the target is the prediction goal.

What this generator cannot do: anything that is not in the margins and the
correlation matrix. Multimodal features and non-monotone dependence get
flattened.
""")

code("""
# The implementation lives in src/copula.py; here I rerun the sanity checks
# on the saved synthetic set to show they hold for the committed file.
from copula import sanity_report
X_train = train[feature_names].values
X_cop = cop[feature_names].values
san = sanity_report(X_cop, cop["target"].values, X_train, feature_names)
print("NaNs:", san["n_nan"], "| Infs:", san["n_inf"])
print("class counts:", san["class_counts"], "| train prevalence:",
      {k: round(v, 3) for k, v in san["train_prevalence"].items()})
print("features with any value outside the train range:", san["out_of_train_range"] or "none")
# Per-class means: does the copula keep the class-conditional locations?
for c in (0, 1):
    d = np.abs(X_train[train["target"].values == c].mean(0) - X_cop[cop["target"].values == c].mean(0))
    print(f"class {c}: max |mean diff| over 30 features = {d.max():.4g}")
""")

md("""
### Generator B: LLM row synthesis

The second generator is Gemini (flash tier), prompted with the schema
(column names, observed ranges, what the target means) plus 8 real example
rows, and asked for batches of 40 new CSV rows. Batches were parsed
defensively: wrong column counts, non-numeric cells, non-0/1 targets, or
values outside the observed ranges (with 2% slack) were rejected and
retried. The rejection log is saved honestly in the results.

The raw API responses are cached under `data/llm_cache/` so the run is
reproducible without new API calls.
""")

code("""
stats = json.load(open(os.path.join(DATA, "llm_stats.json")))
print("batches accepted:", stats["batches_accepted"])
print("batches rejected:", stats["batches_rejected"])
for r in stats["rejections"][:10]:
    print("  rejected batch", r["batch"], "-", r["reason"])
X_llm = llm[feature_names].values
print("\\nLLM class balance:", llm["target"].value_counts(normalize=True).round(3).to_dict())
print("LLM NaNs:", int(llm.isna().sum().sum()))
""")

md("""
## Results""")

md("""
### Fidelity: does the synthetic data look real?

Three checks, from weak to strong:

1. **Per-feature KS tests** (real train vs synthetic). The copula should do
   well here by construction, since it inverts the empirical CDF.
2. **Correlation-matrix distance**: Frobenius norm between the Pearson
   correlation matrices. This tests the dependence structure the copula
   explicitly models.
3. **MMD with an RBF kernel** (median-heuristic bandwidth fixed from the
   real train set so both generators are compared on the same kernel).
   A joint-distribution test that catches things the margins miss.
""")

code("""
from metrics import ks_table, correlation_matrix_distance, mmd_rbf
X_hold = hold[feature_names].values
ks_c = pd.DataFrame(ks_table(X_train, X_cop, feature_names))
ks_l = pd.DataFrame(ks_table(X_train, X_llm, feature_names))
print(f"copula: mean KS={ks_c['ks_stat'].mean():.4f}, max KS={ks_c['ks_stat'].max():.4f}, "
      f"features with p<0.05: {(ks_c['ks_pvalue'] < 0.05).sum()}/30")
print(f"llm:    mean KS={ks_l['ks_stat'].mean():.4f}, max KS={ks_l['ks_stat'].max():.4f}, "
      f"features with p<0.05: {(ks_l['ks_pvalue'] < 0.05).sum()}/30")
print("\\nworst 5 features for the LLM by KS:")
print(ks_l.sort_values("ks_stat", ascending=False)[["feature", "ks_stat", "ks_pvalue", "real_mean", "syn_mean"]].head(5).to_string(index=False))
cd_c = correlation_matrix_distance(X_train, X_cop)
cd_l = correlation_matrix_distance(X_train, X_llm)
mmd_c, gamma = mmd_rbf(X_train, X_cop)
mmd_l, _ = mmd_rbf(X_train, X_llm, gamma=gamma)
print(f"\\ncorrelation-matrix distance: copula={cd_c:.4f}  llm={cd_l:.4f}")
print(f"MMD^2 (gamma={gamma:.4g}): copula={mmd_c:.5f}  llm={mmd_l:.5f}")
""")

md("""
### Utility: train on synthetic, test on real (TSTR)

The practical question: if I train a classifier on synthetic data and
deploy it on real data, how much AUC do I lose versus training on real data?

I train logistic regression and a random forest (300 trees) on each of the
three training sets and score AUC on the same real holdout. The AUC
*differences* use bootstrap 95% CIs (2000 resamples) so the comparison is
honest about sampling noise.
""")

code("""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from metrics import bootstrap_auc_diff
y_hold = hold["target"].values
y_train = train["target"].values
rows = []
for mname, make in [("logreg", lambda: LogisticRegression(max_iter=2000, random_state=SEED)),
                    ("rf", lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]:
    probas = {}
    for key, df in [("real", train), ("copula", cop), ("llm", llm)]:
        m = make().fit(df[feature_names].values, df["target"].values)
        probas[key] = m.predict_proba(X_hold)[:, 1]
    for key in ("real", "copula", "llm"):
        rows.append((mname, key, roc_auc_score(y_hold, probas[key])))
    for key in ("copula", "llm"):
        d = bootstrap_auc_diff(y_hold, probas[key], probas["real"])
        print(f"{mname:6s} {key:6s} vs real: AUC diff {d['auc_diff']:+.4f}  95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
print()
print(pd.DataFrame(rows, columns=["model", "trained_on", "holdout_auc"]).to_string(index=False))
""")

md("""
### Privacy: how close is synthetic to real?

Two checks:

1. **Distance to closest record (DCR).** For each synthetic row, the
   Euclidean distance to the nearest real train row (features standardized
   with the train mean/sd). The baseline is real train rows measured against
   *other* real train rows. If synthetic rows sit closer to the training
   data than real rows sit to each other, the generator is copying.
2. **Membership-inference attack.** A deliberately simple one: use the
   negative nearest-neighbor distance to the synthetic set as a score to
   separate train members from holdout non-members. Attack AUC of 0.5 means
   the synthetic data gives away nothing beyond what any new real row would.
   Anything clearly above 0.5 is leakage.
""")

code("""
from metrics import dcr_stats, membership_inference_auc
dcr_c = dcr_stats(X_train, X_cop, X_hold)
dcr_l = dcr_stats(X_train, X_llm, X_hold)
for name, d in [("copula", dcr_c), ("llm", dcr_l)]:
    print(f"{name:6s} synthetic->train: mean={d['syn_mean']:.3f} median={d['syn_median']:.3f} p5={d['syn_p5']:.3f}")
print(f"       real->real baseline: mean={dcr_c['train_mean']:.3f} median={dcr_c['train_median']:.3f} p5={dcr_c['train_p5']:.3f}")
print(f"       holdout->train     : mean={dcr_c['holdout_mean']:.3f} median={dcr_c['holdout_median']:.3f} p5={dcr_c['holdout_p5']:.3f}")
print(f"\\nMI attack AUC: copula={membership_inference_auc(X_train, X_hold, X_cop):.4f}  "
      f"llm={membership_inference_auc(X_train, X_hold, X_llm):.4f}  (0.5 = no signal)")
""")

md("""
### The tradeoff in one picture

Fidelity (correlation-matrix distance) on x, utility loss (TSTR AUC drop for
the random forest) on y, bubble size proportional to privacy leakage
(MI AUC minus 0.5). The ideal generator would sit at the bottom-left with a
tiny bubble.
""")

code("""
from plotting import fig_tradeoff
res = json.load(open(os.path.join(DATA, "results.json")))
t = res["tstr"]
summary = {
    "copula": {"corr_dist": res["corr_dist_copula"],
               "auc_drop_rf": t["rf_real"]["auc"] - t["rf_copula"]["auc"],
               "mi_auc": res["mi_auc_copula"]},
    "llm": {"corr_dist": res["corr_dist_llm"],
            "auc_drop_rf": t["rf_real"]["auc"] - t["rf_llm"]["auc"],
            "mi_auc": res["mi_auc_llm"]},
}
fig_tradeoff(summary, os.path.join(ROOT, "figures", "tradeoff_notebook.png"))
print("tradeoff figure saved to figures/tradeoff_notebook.png")
print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in summary.items()}, indent=2))
""")

md("""
## Takeaway

- The copula is the fidelity winner by construction: exact margins and an
  explicit correlation fit. Its failure mode is structural: anything not in
  the correlation matrix is lost, and the synthetic rows sit a little too
  close to the training data (higher DCR overlap), which is the price of
  fitting the dependence so tightly.
- The LLM rows are plausible but visibly noisier on the margins: larger KS
  statistics, more rejected p-values, and a bigger correlation-matrix
  distance. The rows spread out more (higher DCR), though the simple
  membership-inference attack finds a touch more signal in the LLM set
  (0.560 vs 0.542 for the copula) -- both barely above the 0.5 no-signal
  line, so I would not call this a privacy win either way.
- For TSTR utility, what matters is whether the classifier's decision
  boundary survives. The numbers above say how much AUC each generator costs
  relative to training on real data, with bootstrap CIs so you can see
  whether the gap is real or noise.
- Practical rule of thumb from this experiment: use the copula when you need
  statistical fidelity and the data will stay in-house. The privacy picture
  is mixed (LLM rows sit further from real records by DCR, but the weak
  membership-inference attack found slightly more signal in them), so I
  would not share either set casually. Neither replaces real data for
  model selection.
""")

nb = nbf.v4.new_notebook()
nb.cells = ROOT_CELLS
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

with open("notebooks/analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/analysis.ipynb with", len(ROOT_CELLS), "cells")
