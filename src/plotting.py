"""Matplotlib figures for the synthetic-data tradeoffs project.

All figures use labeled axes, tight layout, and a restrained style.
Figures are saved by the notebook and the build script.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    }
)

COLORS = {"real": "#2f2f2f", "copula": "#1f77b4", "llm": "#d62728"}


def fig_ks(ks_copula, ks_llm, feature_names, path):
    """KS statistics per feature for both generators (sorted by copula KS)."""
    order = np.argsort([r["ks_stat"] for r in ks_copula])
    names = [feature_names[i] for i in order]
    c_vals = [ks_copula[i]["ks_stat"] for i in order]
    l_vals = [ks_llm[i]["ks_stat"] for i in order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(y + 0.2, c_vals, 0.4, label="Copula", color=COLORS["copula"])
    ax.barh(y - 0.2, l_vals, 0.4, label="LLM", color=COLORS["llm"])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("KS statistic (lower means margins match better)")
    ax.set_title("Per-feature margin fidelity: two-sample KS statistic")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def fig_dcr(dcr_copula, dcr_llm, path):
    """DCR distributions: synthetic-to-train vs real train-to-train."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dcr, name, color in [
        (axes[0], dcr_copula, "Copula", COLORS["copula"]),
        (axes[1], dcr_llm, "LLM", COLORS["llm"]),
    ]:
        syn = dcr["syn_all"]
        base = dcr["train_all"]
        bins = np.histogram_bin_edges(np.concatenate([syn, base]), bins=40)
        ax.hist(base, bins=bins, alpha=0.5, label="real-to-real baseline",
                color=COLORS["real"], density=True)
        ax.hist(syn, bins=bins, alpha=0.6, label="synthetic-to-real",
                color=color, density=True)
        ax.set_xlabel("Nearest-neighbor distance (standardized features)")
        ax.set_title(f"{name}: distance to closest real train record")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.suptitle("How close are synthetic rows to real training rows?")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_tradeoff(summary, path):
    """3-way tradeoff: fidelity (corr distance) vs utility (TSTR AUC drop)
    with point size showing privacy leakage (MI AUC - 0.5)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, color in [("copula", COLORS["copula"]), ("llm", COLORS["llm"])]:
        s = summary[key]
        size = 200 + 4000 * max(s["mi_auc"] - 0.5, 0.0)
        ax.scatter(
            s["corr_dist"],
            s["auc_drop_rf"],
            s=size,
            color=color,
            alpha=0.6,
            label=f"{key} (MI AUC {s['mi_auc']:.3f})",
            edgecolors="black",
        )
        ax.annotate(
            key, (s["corr_dist"], s["auc_drop_rf"]),
            textcoords="offset points", xytext=(8, -8), fontsize=11,
        )
    ax.set_xlabel("Fidelity: correlation-matrix distance (lower is better)")
    ax.set_ylabel("Utility: TSTR AUC drop vs train-on-real, RF (lower is better)")
    ax.set_title("Fidelity vs utility, bubble size = privacy leakage")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def fig_forest(estimates, path):
    """Forest plot of causal-style comparison of TSTR AUCs with bootstrap CIs."""
    labels = list(estimates.keys())
    vals = [estimates[k]["auc"] for k in labels]
    los = [estimates[k]["ci_low"] for k in labels]
    his = [estimates[k]["ci_high"] for k in labels]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, k in enumerate(labels):
        color = COLORS.get(k.split()[0].lower(), "#2f2f2f")
        ax.errorbar(vals[i], y[i], xerr=[[vals[i] - los[i]], [his[i] - vals[i]]],
                    fmt="o", color=color, ecolor="black", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUC on real holdout (95% bootstrap CI)")
    ax.set_title("TSTR utility: model trained on synthetic vs on real data")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    fig.savefig(path)
    plt.close(fig)
