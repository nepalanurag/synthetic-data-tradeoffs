"""Gaussian copula synthetic data generator, implemented by hand.

Method: for each target class separately
  1. Rank-transform each feature to uniform on (0, 1) using the empirical CDF
     (average ranks, divided by n + 1 so no exact 0 or 1).
  2. Map uniform values to standard normals with the probit (ndtri).
  3. Estimate the mean and covariance of the latent normals.
  4. Sample new latent normals, map back through the normal CDF, then through
     the empirical inverse CDF of each feature (linear interpolation).

Class labels are sampled from the training prevalence. This per-class fit is
a choice: it guarantees the class-conditional structure and the target
prevalence are reproduced, at the cost of not modeling the joint
feature-target distribution as one block. For a classification utility test
(TSTR) that is the right trade-off.

Why a Gaussian copula: it reproduces every univariate margin exactly (up to
sampling noise) and the rank correlation structure, without assuming the
margins are Gaussian. It cannot reproduce structure that is not in the
correlation matrix, for example multimodality or non-monotone dependence.
"""

import numpy as np
from scipy.stats import norm


class GaussianCopula:
    def __init__(self, seed=20261002):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._fitted = False

    def fit(self, X, y):
        """Fit one copula per class.

        Parameters
        ----------
        X : (n, p) ndarray of continuous features
        y : (n,) ndarray of class labels
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf; clean before fitting")
        self.classes_ = np.unique(y)
        self.class_probs_ = np.array([np.mean(y == c) for c in self.classes_])
        self.models_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            n, p = Xc.shape
            # Rank to uniform via empirical CDF. n+1 avoids exact 0/1 so the
            # probit below stays finite.
            ranks = np.apply_along_axis(
                lambda col: np.argsort(np.argsort(col)).astype(float) + 1.0, 0, Xc
            )
            u = ranks / (n + 1.0)
            z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
            mean = z.mean(axis=0)
            cov = np.cov(z, rowvar=False)
            # Guard against a numerically singular covariance (constant or
            # near-constant features).
            cov = cov + 1e-6 * np.eye(p)
            # Store sorted values per feature for the inverse-CDF step.
            sorted_x = np.sort(Xc, axis=0)
            self.models_[c] = {
                "n": n,
                "mean": mean,
                "cov": cov,
                "sorted_x": sorted_x,
                "x_min": Xc.min(axis=0),
                "x_max": Xc.max(axis=0),
            }
        self._fitted = True
        return self

    def _inverse_cdf(self, u_col, sorted_col):
        # Piecewise-linear inverse empirical CDF: quantile(u) of the sorted data.
        n = len(sorted_col)
        grid = (np.arange(n) + 1.0) / (n + 1.0)
        return np.interp(np.clip(u_col, 0.0, 1.0), grid, sorted_col)

    def sample(self, n_rows):
        """Draw n_rows synthetic (X, y)."""
        if not self._fitted:
            raise RuntimeError("fit() must be called before sample()")
        rng = self.rng
        y_syn = rng.choice(self.classes_, size=n_rows, p=self.class_probs_)
        X_syn = np.zeros((n_rows, len(self.models_[self.classes_[0]]["mean"])))
        for c, m in self.models_.items():
            idx = np.flatnonzero(y_syn == c)
            if idx.size == 0:
                continue
            z = rng.multivariate_normal(m["mean"], m["cov"], size=idx.size)
            u = norm.cdf(z)
            for j in range(z.shape[1]):
                X_syn[idx, j] = self._inverse_cdf(u[:, j], m["sorted_x"][:, j])
        return X_syn, y_syn


def sanity_report(X_syn, y_syn, X_train, feature_names):
    """Return a dict of simple sanity checks: NaNs, range violations, prevalence."""
    X_syn = np.asarray(X_syn)
    report = {
        "n_rows": int(X_syn.shape[0]),
        "n_cols": int(X_syn.shape[1]),
        "n_nan": int(np.isnan(X_syn).sum()),
        "n_inf": int(np.isinf(X_syn).sum()),
        "class_counts": {str(k): int((y_syn == k).sum()) for k in np.unique(y_syn)},
        "train_prevalence": {
            str(k): float(np.mean(y_syn == k)) for k in np.unique(y_syn)
        },
        "out_of_train_range": {},
    }
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    for j, name in enumerate(feature_names):
        n_low = int((X_syn[:, j] < x_min[j]).sum())
        n_high = int((X_syn[:, j] > x_max[j]).sum())
        if n_low or n_high:
            report["out_of_train_range"][name] = {"below": n_low, "above": n_high}
    return report
