"""Create the base data files: train/holdout split and copula synthetic set.

Deterministic (seed 20261002). No API calls. Run before the LLM synthesis
finishes; run_eval.py consumes these plus the LLM outputs.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copula import GaussianCopula, sanity_report

SEED = 20261002
N_SYN = 500
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)


def main():
    ds = load_breast_cancer()
    X = ds.data.astype(float)
    y = ds.target.astype(int)
    feature_names = list(ds.feature_names)
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    cop = GaussianCopula(seed=SEED).fit(X_train, y_train)
    X_cop, y_cop = cop.sample(N_SYN)
    san = sanity_report(X_cop, y_cop, X_train, feature_names)
    assert san["n_nan"] == 0 and san["n_inf"] == 0, san
    assert not san["out_of_train_range"], san["out_of_train_range"]
    cols = ["target"] + feature_names
    pd.DataFrame(np.column_stack([y_train, X_train]), columns=cols).to_csv(
        os.path.join(DATA, "train.csv"), index=False)
    pd.DataFrame(np.column_stack([y_hold, X_hold]), columns=cols).to_csv(
        os.path.join(DATA, "holdout.csv"), index=False)
    pd.DataFrame(np.column_stack([y_cop, X_cop]), columns=cols).to_csv(
        os.path.join(DATA, "copula_synthetic.csv"), index=False)
    print("wrote train/holdout/copula CSVs; sanity:", {
        "n_nan": san["n_nan"], "class_counts": san["class_counts"]}, flush=True)


if __name__ == "__main__":
    main()
