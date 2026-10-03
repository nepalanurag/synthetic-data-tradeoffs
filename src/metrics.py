"""Evaluation metrics for the fidelity / utility / privacy comparison.

Fidelity
  - per-column two-sample Kolmogorov-Smirnov statistic and p-value
  - correlation-matrix distance: Frobenius norm of the difference between the
    Pearson correlation matrices of real and synthetic data
  - MMD with an RBF kernel, bandwidth set by the median heuristic

Utility (TSTR)
  - train logistic regression and random forest on synthetic data, score AUC
    on the real holdout; baseline trains the same models on the real train
    split. AUC differences get bootstrap 95% CIs.

Privacy
  - distance-to-closest-record (DCR): for each synthetic row, distance to its
    nearest real-train row (features standardized with the train mean/sd);
    the baseline is real-train rows measured against the other real-train
    rows. If synthetic rows hug the training data more than real rows hug
    each other, the generator is copying.
  - membership-inference attack: a threshold on nearest-neighbor distance to
    the synthetic set, separating train members from holdout non-members.
    Attack AUC of 0.5 means no leakage signal.
"""

import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


def ks_table(X_real, X_syn, feature_names):
    rows = []
    for j, name in enumerate(feature_names):
        stat, pval = ks_2samp(X_real[:, j], X_syn[:, j])
        rows.append(
            {
                "feature": name,
                "ks_stat": float(stat),
                "ks_pvalue": float(pval),
                "real_mean": float(X_real[:, j].mean()),
                "syn_mean": float(X_syn[:, j].mean()),
                "real_std": float(X_real[:, j].std()),
                "syn_std": float(X_syn[:, j].std()),
            }
        )
    return rows


def correlation_matrix_distance(X_real, X_syn):
    cr = np.corrcoef(X_real, rowvar=False)
    cs = np.corrcoef(X_syn, rowvar=False)
    return float(np.linalg.norm(cr - cs, ord="fro"))


def mmd_rbf(X, Y, gamma=None):
    """Squared MMD with an RBF kernel between samples X and Y."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        Z = np.vstack([X, Y])
        d = cdist(Z, Z, metric="euclidean")
        med = np.median(d[d > 0])
        gamma = 1.0 / (2.0 * med**2) if med > 0 else 1.0

    def k(A, B):
        return np.exp(-gamma * cdist(A, B, metric="sqeuclidean"))

    n, m = len(X), len(Y)
    kxx = k(X, X)
    kyy = k(Y, Y)
    kxy = k(X, Y)
    # Unbiased-ish squared MMD, dropping the diagonal self-terms.
    mmd2 = (
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )
    return float(max(mmd2, 0.0)), float(gamma)


def _standardize(X_train, *arrays):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return [((a - mu) / sd) for a in arrays]


def dcr_stats(X_train, X_syn, X_holdout=None):
    """Nearest-neighbor distances from synthetic rows to real train rows.

    Baseline: each real-train row's distance to its nearest *other* real-train
    row (leave-one-out), and optionally real holdout rows to train.
    """
    Xs_train, Xs_syn = _standardize(X_train, X_train, X_syn)
    d_syn = cdist(Xs_syn, Xs_train).min(axis=1)
    d_all = cdist(Xs_train, Xs_train)
    np.fill_diagonal(d_all, np.inf)
    d_train = d_all.min(axis=1)
    out = {
        "syn_mean": float(d_syn.mean()),
        "syn_median": float(np.median(d_syn)),
        "syn_p5": float(np.percentile(d_syn, 5)),
        "train_mean": float(d_train.mean()),
        "train_median": float(np.median(d_train)),
        "train_p5": float(np.percentile(d_train, 5)),
        "syn_all": d_syn,
        "train_all": d_train,
    }
    if X_holdout is not None:
        (Xs_hold,) = _standardize(X_train, X_holdout)
        d_hold = cdist(Xs_hold, Xs_train).min(axis=1)
        out["holdout_mean"] = float(d_hold.mean())
        out["holdout_median"] = float(np.median(d_hold))
        out["holdout_p5"] = float(np.percentile(d_hold, 5))
    return out


def membership_inference_auc(X_train, X_holdout, X_syn):
    """Simple MI attack: low NN distance to the synthetic set -> predicted member.

    Members = real train rows (label 1), non-members = real holdout rows
    (label 0). Score = negative nearest-neighbor distance to the synthetic set.
    """
    Xs_train, Xs_hold, Xs_syn = _standardize(X_train, X_train, X_holdout, X_syn)
    d_mem = cdist(Xs_train, Xs_syn).min(axis=1)
    d_non = cdist(Xs_hold, Xs_syn).min(axis=1)
    scores = np.concatenate([-d_mem, -d_non])
    labels = np.concatenate([np.ones_like(d_mem), np.zeros_like(d_non)])
    auc = roc_auc_score(labels, scores)
    return float(auc)


def bootstrap_auc_ci(y_true, scores, n_boot=2000, seed=20261002):
    """Bootstrap 95% CI for a single AUC on fixed test labels."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], scores[idx])
    aucs = aucs[~np.isnan(aucs)]
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
    }


def bootstrap_auc_diff(y_true, scores_a, scores_b, n_boot=2000, seed=20261002):
    """Bootstrap 95% CI for AUC(a) - AUC(b) on the same test labels."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    base = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    diffs = np.empty(n_boot)
    n = len(y_true)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc_score(y_true[idx], scores_a[idx]) - roc_auc_score(
            y_true[idx], scores_b[idx]
        )
    diffs = diffs[~np.isnan(diffs)]
    return {
        "auc_diff": float(base),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
