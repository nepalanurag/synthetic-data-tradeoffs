"""Build dashboard.html: one self-contained Plotly dashboard.

Reads data/results.json, data/ks_*.csv, data/*.csv. Plotly comes from a
pinned CDN; all data is embedded in the file (no external data fetches).
Sections: Overview, Methods, Results, Limitations.
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def main():
    res = json.load(open(os.path.join(DATA, "results.json")))
    t = res["tstr"]
    ks_c = pd.read_csv(os.path.join(DATA, "ks_copula.csv"))
    ks_l = pd.read_csv(os.path.join(DATA, "ks_llm.csv"))
    feats = ks_c["feature"].tolist()

    def fmt(x, nd=4):
        return f"{x:.{nd}f}"

    # ---- data for plots (embedded as JSON) ----
    ks_data = {
        "features": feats,
        "copula": [round(v, 4) for v in ks_c["ks_stat"]],
        "llm": [round(v, 4) for v in ks_l["ks_stat"]],
    }
    dcr = res["dcr_copula"], res["dcr_llm"]
    tradeoff = {
        "copula": {
            "corr": round(res["corr_dist_copula"], 4),
            "drop": round(t["rf_real"]["auc"] - t["rf_copula"]["auc"], 4),
            "mi": round(res["mi_auc_copula"], 4),
        },
        "llm": {
            "corr": round(res["corr_dist_llm"], 4),
            "drop": round(t["rf_real"]["auc"] - t["rf_llm"]["auc"], 4),
            "mi": round(res["mi_auc_llm"], 4),
        },
    }

    tstr_rows = []
    for mname, mlabel in [("rf", "Random forest"), ("logreg", "Logistic regression")]:
        for key, klabel in [("real", "Real train"), ("copula", "Copula"),
                            ("llm", "LLM")]:
            r = t[f"{mname}_{key}"]
            tstr_rows.append({
                "model": mlabel, "trained_on": klabel,
                "auc": round(r["auc"], 4),
                "lo": round(r["ci_low"], 4), "hi": round(r["ci_high"], 4),
            })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic data tradeoffs: fidelity, utility, privacy</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 960px; margin: 0 auto; padding: 24px; color: #222; line-height: 1.55; }}
  h1 {{ font-size: 1.7em; }} h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 44px; }}
  table {{ border-collapse: collapse; margin: 12px 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #f4f4f4; }} td:first-child, th:first-child {{ text-align: left; }}
  .plot {{ width: 100%; height: 460px; }}
  .kpi {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .kpi div {{ background: #f7f7f7; border: 1px solid #ddd; padding: 10px 16px; border-radius: 6px; }}
  .kpi b {{ font-size: 1.25em; display: block; }}
  nav a {{ margin-right: 18px; }}
  code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }}
</style>
</head>
<body>
<nav><a href="#overview">Overview</a><a href="#methods">Methods</a>
<a href="#results">Results</a><a href="#limitations">Limitations</a></nav>

<h1>Synthetic tabular data: the fidelity-utility-privacy tradeoff</h1>
<p>Two generators, one dataset (breast cancer Wisconsin, 569 rows, 30 numeric
features), three axes of evaluation. The copula is fit on 398 training rows;
the LLM (Gemini flash) saw the schema plus 8 example rows. 500 synthetic rows
from the copula, 126 validated rows from the LLM (21 batches accepted, 5
rejected). A fixed 171-row holdout, never shown to either generator, anchors the
utility and privacy tests. Seed 20261002 throughout.</p>

<h2 id="overview">Overview</h2>
<div class="kpi">
  <div><b>{fmt(res["corr_dist_copula"])} vs {fmt(res["corr_dist_llm"])}</b>correlation-matrix distance (copula vs LLM; lower is better)</div>
  <div><b>{fmt(tradeoff["copula"]["drop"], 3)} vs {fmt(tradeoff["llm"]["drop"], 3)}</b>random-forest TSTR AUC drop vs training on real data</div>
  <div><b>{fmt(res["mi_auc_copula"], 3)} vs {fmt(res["mi_auc_llm"], 3)}</b>membership-inference attack AUC (0.5 = no leakage)</div>
</div>
<p>Bottom line: the copula wins on fidelity by construction and its TSTR utility
gap vs training on real data is within noise; the LLM rows are noisier on the
margins and cost real AUC (-0.0303, CI fully below zero). Privacy is a wash:
LLM rows sit further from real records by DCR, but the weak
membership-inference attack found slightly more signal in them (0.560 vs
0.542). Neither is a free lunch.</p>

<h2 id="methods">Methods</h2>
<h3>Generator A: Gaussian copula (implemented by hand)</h3>
<p>Rank-transform each feature to uniform via the empirical CDF, map to
standard normals with the probit, fit mean and covariance, sample, and map
back through the inverse empirical CDF. Fit separately per class; class
labels sampled from the training prevalence. This reproduces every margin
exactly and the rank correlations, but nothing beyond the correlation
matrix.</p>
<h3>Generator B: LLM row synthesis</h3>
<p>Gemini flash prompted with column names, observed ranges, the target
meaning, and 8 real example rows; asked for batches of 20 CSV rows. Batches
were validated (column count, numeric cells, 0/1 target, in-range values)
and rejected batches were retried: {res["llm_batches_accepted"]} accepted,
{res["llm_batches_rejected"]} rejected. Raw responses are cached in the repo
so the run is reproducible.</p>
<h3>Evaluation</h3>
<ul>
<li><b>Fidelity:</b> per-feature two-sample KS tests; correlation-matrix
distance (Frobenius norm); MMD with RBF kernel, bandwidth from the median
heuristic on the real train set (gamma={res["mmd_gamma"]:.4g}).</li>
<li><b>Utility (TSTR):</b> logistic regression and random forest trained on
each synthetic set, AUC on the real holdout; train-on-real baseline; AUC
differences with bootstrap 95% CIs (2000 resamples).</li>
<li><b>Privacy:</b> distance-to-closest-record distributions
(synthetic-to-train vs real-to-real baseline); a threshold membership-inference
attack on nearest-neighbor distance to the synthetic set.</li>
</ul>

<h2 id="results">Results</h2>
<h3>Fidelity: per-feature KS statistics</h3>
<div id="ksplot" class="plot"></div>
<p>Mean KS: copula {fmt(res["ks_copula_mean"])} (max {fmt(res["ks_copula_max"])},
{res["ks_copula_p_below_05"]}/30 features with p &lt; 0.05); LLM
{fmt(res["ks_llm_mean"])} (max {fmt(res["ks_llm_max"])},
{res["ks_llm_p_below_05"]}/30 with p &lt; 0.05).</p>
<h3>Fidelity: joint structure</h3>
<table>
<tr><th></th><th>Correlation-matrix distance</th><th>MMD^2 (RBF)</th></tr>
<tr><td>Copula</td><td>{fmt(res["corr_dist_copula"])}</td><td>{fmt(res["mmd_copula"], 5)}</td></tr>
<tr><td>LLM</td><td>{fmt(res["corr_dist_llm"])}</td><td>{fmt(res["mmd_llm"], 5)}</td></tr>
</table>
<h3>Utility: train on synthetic, test on real</h3>
<div id="tstrplot" class="plot"></div>
<table>
<tr><th>Model</th><th>Trained on</th><th>Holdout AUC</th><th>95% CI</th></tr>
{''.join(f"<tr><td>{r['model']}</td><td>{r['trained_on']}</td><td>{r['auc']:.4f}</td><td>[{r['lo']:.4f}, {r['hi']:.4f}]</td></tr>" for r in tstr_rows)}
</table>
<h3>Privacy</h3>
<table>
<tr><th></th><th>DCR median (synthetic to train)</th><th>DCR median (real to real)</th><th>MI attack AUC</th></tr>
<tr><td>Copula</td><td>{fmt(res["dcr_copula"]["syn_median"], 3)}</td><td>{fmt(res["dcr_copula"]["train_median"], 3)}</td><td>{fmt(res["mi_auc_copula"], 3)}</td></tr>
<tr><td>LLM</td><td>{fmt(res["dcr_llm"]["syn_median"], 3)}</td><td>{fmt(res["dcr_llm"]["train_median"], 3)}</td><td>{fmt(res["mi_auc_llm"], 3)}</td></tr>
</table>
<p>Real holdout rows sit at median distance {fmt(res["dcr_copula"]["holdout_median"], 3)}
from the train set; synthetic rows closer than that are hugging the training data.</p>
<h3>The 3-way tradeoff</h3>
<div id="tradeplot" class="plot"></div>

<h2 id="limitations">Limitations</h2>
<ul>
<li>One dataset, fully numeric, 569 rows. Results may not carry to mixed-type,
high-dimensional, or heavy-tailed data.</li>
<li>The membership-inference attack is deliberately simple (distance
threshold). A stronger attack could find more leakage; these numbers are a
lower bound on risk, not a certificate of safety.</li>
<li>LLM synthesis depends on the model version and prompt; the raw responses
are cached in the repo so this exact run is reproducible, but a rerun with a
different model may differ.</li>
<li>TSTR measures one downstream task (binary classification AUC). A different
task could rank the generators differently.</li>
<li>The copula is fit per class, which helps classification utility but would
be the wrong choice if the target itself needed joint modeling.</li>
</ul>
<p>Code, data, and the executed analysis notebook are in the
<a href="https://github.com/nepalanurag/synthetic-data-tradeoffs">GitHub repo</a>.
Seed 20261002 everywhere; no numbers on this page are invented.</p>

<script>
const KS = {json.dumps(ks_data)};
Plotly.newPlot("ksplot", [
  {{y: KS.features, x: KS.copula, type: "bar", orientation: "h", name: "Copula", marker: {{color: "#1f77b4"}}}},
  {{y: KS.features, x: KS.llm, type: "bar", orientation: "h", name: "LLM", marker: {{color: "#d62728"}}}}
], {{title: "Per-feature KS statistic (lower = margins match better)",
    xaxis: {{title: "KS statistic"}}, barmode: "group", height: 560,
    margin: {{l: 190}}}});

const TSTR = {json.dumps(tstr_rows)};
{{
  const models = [...new Set(TSTR.map(r => r.model))];
  const traces = models.map((m, i) => {{
    const rs = TSTR.filter(r => r.model === m);
    return {{y: rs.map(r => r.trained_on), x: rs.map(r => r.auc),
      error_x: {{type: "data",
        array: rs.map(r => r.hi - r.auc), arrayminus: rs.map(r => r.auc - r.lo)}},
      type: "scatter", mode: "markers", name: m,
      marker: {{size: 10, color: ["#1f77b4", "#d62728"][i]}}}};
  }});
  Plotly.newPlot("tstrplot", traces,
    {{title: "TSTR: AUC on the real holdout (95% bootstrap CI)",
      xaxis: {{title: "AUC", range: [0.9, 1.0]}}}});
}}

const TO = {json.dumps(tradeoff)};
Plotly.newPlot("tradeplot", [
  {{x: [TO.copula.corr], y: [TO.copula.drop], mode: "markers+text", name: "Copula",
    text: ["Copula (MI AUC " + TO.copula.mi + ")"], textposition: "top center",
    marker: {{size: 18 + 400 * Math.max(TO.copula.mi - 0.5, 0), color: "#1f77b4"}}}},
  {{x: [TO.llm.corr], y: [TO.llm.drop], mode: "markers+text", name: "LLM",
    text: ["LLM (MI AUC " + TO.llm.mi + ")"], textposition: "top center",
    marker: {{size: 18 + 400 * Math.max(TO.llm.mi - 0.5, 0), color: "#d62728"}}}}
], {{title: "Fidelity vs utility; bubble size = privacy leakage (MI AUC - 0.5)",
    xaxis: {{title: "Correlation-matrix distance (lower is better)"}},
    yaxis: {{title: "TSTR AUC drop vs train-on-real, RF (lower is better)"}}}});
</script>
</body>
</html>
"""
    out = os.path.join(ROOT, "dashboard.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, f"({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
