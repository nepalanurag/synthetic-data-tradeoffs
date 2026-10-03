"""Evaluate from saved data files: metrics, results.json, figures.

Expects in data/: train.csv, holdout.csv, copula_synthetic.csv,
X_llm.npy, y_llm.npy, llm_stats.json.
Writes data/results.json, data/ks_*.csv, data/llm_synthetic.csv, figures/*.png.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (
    ks_table,
    correlation_matrix_distance,
    mmd_rbf,
    dcr_stats,
    membership_inference_auc,
    bootstrap_auc_ci,
    bootstrap_auc_diff,
)
from plotting import fig_ks, fig_dcr, fig_tradeoff, fig_forest

SEED = 20261002
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    hold = pd.read_csv(os.path.join(DATA, "holdout.csv"))
    cop = pd.read_csv(os.path.join(DATA, "copula_synthetic.csv"))
    feature_names = [c for c in train.columns if c != "target"]
    X_train = train[feature_names].values
    y_train = train["target"].values.astype(int)
    X_hold = hold[feature_names].values
    y_hold = hold["target"].values.astype(int)
    X_cop = cop[feature_names].values
    y_cop = cop["target"].values.astype(int)
    X_llm = np.load(os.path.join(DATA, "X_llm.npy"))
    y_llm = np.load(os.path.join(DATA, "y_llm.npy")).astype(int)
    llm_stats = json.load(open(os.path.join(DATA, "llm_stats.json")))
    pd.DataFrame(np.column_stack([y_llm, X_llm]),
                 columns=["target"] + feature_names).to_csv(
        os.path.join(DATA, "llm_synthetic.csv"), index=False)

    results = {"seed": SEED,
               "train_n": len(X_train), "holdout_n": len(X_hold),
               "n_syn_copula": len(X_cop), "n_syn_llm": len(X_llm),
               "llm_rejections": llm_stats["rejections"],
               "llm_batches_accepted": llm_stats["batches_accepted"],
               "llm_batches_rejected": llm_stats["batches_rejected"],
               "llm_quarantined": llm_stats.get("quarantined", [])}

    ks_c = ks_table(X_train, X_cop, feature_names)
    ks_l = ks_table(X_train, X_llm, feature_names)
    pd.DataFrame(ks_c).to_csv(os.path.join(DATA, "ks_copula.csv"), index=False)
    pd.DataFrame(ks_l).to_csv(os.path.join(DATA, "ks_llm.csv"), index=False)
    results["ks_copula_mean"] = float(np.mean([r["ks_stat"] for r in ks_c]))
    results["ks_copula_max"] = float(np.max([r["ks_stat"] for r in ks_c]))
    results["ks_copula_p_below_05"] = int(sum(r["ks_pvalue"] < 0.05 for r in ks_c))
    results["ks_llm_mean"] = float(np.mean([r["ks_stat"] for r in ks_l]))
    results["ks_llm_max"] = float(np.max([r["ks_stat"] for r in ks_l]))
    results["ks_llm_p_below_05"] = int(sum(r["ks_pvalue"] < 0.05 for r in ks_l))
    results["corr_dist_copula"] = correlation_matrix_distance(X_train, X_cop)
    results["corr_dist_llm"] = correlation_matrix_distance(X_train, X_llm)
    mmd_c, gamma = mmd_rbf(X_train, X_cop)
    mmd_l, _ = mmd_rbf(X_train, X_llm, gamma=gamma)
    results["mmd_gamma"] = gamma
    results["mmd_copula"] = mmd_c
    results["mmd_llm"] = mmd_l

    models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED,
                                     n_jobs=-1),
    }
    tstr = {}
    proba = {}
    for name, proto in models.items():
        for key, (Xs, ys) in {"real": (X_train, y_train),
                              "copula": (X_cop, y_cop),
                              "llm": (X_llm, y_llm)}.items():
            m = proto.__class__(**proto.get_params())
            m.fit(Xs, ys)
            p = m.predict_proba(X_hold)[:, 1]
            proba[(name, key)] = p
            tstr[f"{name}_{key}"] = bootstrap_auc_ci(y_hold, p)
        tstr[f"{name}_copula_vs_real"] = bootstrap_auc_diff(
            y_hold, proba[(name, "copula")], proba[(name, "real")])
        tstr[f"{name}_llm_vs_real"] = bootstrap_auc_diff(
            y_hold, proba[(name, "llm")], proba[(name, "real")])
    results["tstr"] = tstr

    dcr_c = dcr_stats(X_train, X_cop, X_hold)
    dcr_l = dcr_stats(X_train, X_llm, X_hold)
    results["dcr_copula"] = {k: v for k, v in dcr_c.items()
                             if not k.endswith("_all")}
    results["dcr_llm"] = {k: v for k, v in dcr_l.items() if not k.endswith("_all")}
    results["mi_auc_copula"] = membership_inference_auc(X_train, X_hold, X_cop)
    results["mi_auc_llm"] = membership_inference_auc(X_train, X_hold, X_llm)

    with open(os.path.join(DATA, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig_ks(ks_c, ks_l, feature_names, os.path.join(FIG, "ks_per_feature.png"))
    fig_dcr(dcr_c, dcr_l, os.path.join(FIG, "dcr_distributions.png"))
    summary = {
        "copula": {"corr_dist": results["corr_dist_copula"],
                   "auc_drop_rf": tstr["rf_real"]["auc"] - tstr["rf_copula"]["auc"],
                   "mi_auc": results["mi_auc_copula"]},
        "llm": {"corr_dist": results["corr_dist_llm"],
                "auc_drop_rf": tstr["rf_real"]["auc"] - tstr["rf_llm"]["auc"],
                "mi_auc": results["mi_auc_llm"]},
    }
    fig_tradeoff(summary, os.path.join(FIG, "tradeoff.png"))
    fig_forest({
        "Real train (baseline)": tstr["rf_real"],
        "Copula synthetic": tstr["rf_copula"],
        "LLM synthetic": tstr["rf_llm"],
        "Real train (baseline) ": tstr["logreg_real"],
        "Copula synthetic ": tstr["logreg_copula"],
        "LLM synthetic ": tstr["logreg_llm"],
    }, os.path.join(FIG, "tstr_forest.png"))

    print(json.dumps({
        "ks_mean": [results["ks_copula_mean"], results["ks_llm_mean"]],
        "corr_dist": [results["corr_dist_copula"], results["corr_dist_llm"]],
        "mmd": [results["mmd_copula"], results["mmd_llm"]],
        "tstr_rf": [tstr["rf_real"]["auc"], tstr["rf_copula"]["auc"],
                    tstr["rf_llm"]["auc"]],
        "tstr_lr": [tstr["logreg_real"]["auc"], tstr["logreg_copula"]["auc"],
                    tstr["logreg_llm"]["auc"]],
        "mi_auc": [results["mi_auc_copula"], results["mi_auc_llm"]],
        "dcr_syn_median": [dcr_c["syn_median"], dcr_l["syn_median"]],
        "dcr_train_median": dcr_c["train_median"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
