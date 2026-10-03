"""Fill @@PLACEHOLDER@@ tokens in README.md and REPORT.md from data/results.json.

Every number in the user-facing docs comes from the computed results file.
Run after src/run_all.py (or the resume pipeline) has written results.json.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def f(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    res = json.load(open(os.path.join(DATA, "results.json")))
    t = res["tstr"]
    vals = {
        "KS_COPULA_MEAN": f(res["ks_copula_mean"]),
        "KS_COPULA_MAX": f(res["ks_copula_max"]),
        "KS_COPULA_P": str(res["ks_copula_p_below_05"]),
        "KS_LLM_MEAN": f(res["ks_llm_mean"]),
        "KS_LLM_MAX": f(res["ks_llm_max"]),
        "KS_LLM_P": str(res["ks_llm_p_below_05"]),
        "CORR_COPULA": f(res["corr_dist_copula"]),
        "CORR_LLM": f(res["corr_dist_llm"]),
        "MMD_COPULA": f(res["mmd_copula"], 5),
        "MMD_LLM": f(res["mmd_llm"], 5),
        "MMD_GAMMA": f"{res['mmd_gamma']:.2e}",
        "TSTR_RF_REAL": f(t["rf_real"]["auc"]),
        "TSTR_RF_REAL_LO": f(t["rf_real"]["ci_low"]),
        "TSTR_RF_REAL_HI": f(t["rf_real"]["ci_high"]),
        "TSTR_RF_COPULA": f(t["rf_copula"]["auc"]),
        "TSTR_RF_COPULA_LO": f(t["rf_copula"]["ci_low"]),
        "TSTR_RF_COPULA_HI": f(t["rf_copula"]["ci_high"]),
        "TSTR_RF_LLM": f(t["rf_llm"]["auc"]),
        "TSTR_RF_LLM_LO": f(t["rf_llm"]["ci_low"]),
        "TSTR_RF_LLM_HI": f(t["rf_llm"]["ci_high"]),
        "TSTR_LR_REAL": f(t["logreg_real"]["auc"]),
        "TSTR_LR_REAL_LO": f(t["logreg_real"]["ci_low"]),
        "TSTR_LR_REAL_HI": f(t["logreg_real"]["ci_high"]),
        "TSTR_LR_COPULA": f(t["logreg_copula"]["auc"]),
        "TSTR_LR_COPULA_LO": f(t["logreg_copula"]["ci_low"]),
        "TSTR_LR_COPULA_HI": f(t["logreg_copula"]["ci_high"]),
        "TSTR_LR_LLM": f(t["logreg_llm"]["auc"]),
        "TSTR_LR_LLM_LO": f(t["logreg_llm"]["ci_low"]),
        "TSTR_LR_LLM_HI": f(t["logreg_llm"]["ci_high"]),
        "DROP_COPULA": f"{t['rf_real']['auc'] - t['rf_copula']['auc']:+.4f}",
        "DROP_LLM": f"{t['rf_real']['auc'] - t['rf_llm']['auc']:+.4f}",
        "DCR_COPULA": f(res["dcr_copula"]["syn_median"], 3),
        "DCR_LLM": f(res["dcr_llm"]["syn_median"], 3),
        "DCR_BASE": f(res["dcr_copula"]["train_median"], 3),
        "DCR_HOLDOUT": f(res["dcr_copula"]["holdout_median"], 3),
        "MI_COPULA": f(res["mi_auc_copula"], 3),
        "MI_LLM": f(res["mi_auc_llm"], 3),
        "LLM_ACCEPTED": str(res["llm_batches_accepted"]),
        "LLM_REJECTED": str(res["llm_batches_rejected"]),
    }
    for fname in ["README.md", "REPORT.md"]:
        path = os.path.join(ROOT, fname)
        text = open(path).read()
        missing = []

        def repl(m):
            key = m.group(1)
            if key not in vals:
                missing.append(key)
                return m.group(0)
            return vals[key]

        text = re.sub(r"@@([A-Z_]+)@@", repl, text)
        if missing:
            raise SystemExit(f"{fname}: unfilled tokens {missing}")
        open(path, "w").write(text)
        print(f"filled {fname}")
    # Double-check nothing is left.
    for fname in ["README.md", "REPORT.md"]:
        text = open(os.path.join(ROOT, fname)).read()
        assert "@@" not in text, f"leftover token in {fname}"
    print("all placeholders filled")


if __name__ == "__main__":
    main()
