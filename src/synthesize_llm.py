"""LLM row synthesis step: generate 500 validated synthetic rows via Gemini.

Reads data/train.csv for the schema, ranges, and example rows. Raw API
responses are cached in data/llm_cache/ (append-only; reruns reuse the cache
without new API calls). Writes data/X_llm.npy, data/y_llm.npy,
data/llm_stats.json.

Auth uses the stored connector credential; only
generativelanguage.googleapis.com is contacted.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_synth import synthesize, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    feature_names = [c for c in train.columns if c != "target"]
    X_train = train[feature_names].values
    y_train = train["target"].values.astype(int)
    feature_ranges = {
        n: (float(X_train[:, j].min()), float(X_train[:, j].max()))
        for j, n in enumerate(feature_names)
    }
    target_desc = (
        "target is 0 for a malignant tumor and 1 for a benign tumor, from "
        "cell-nucleus measurements of breast mass images."
    )
    t0 = time.time()
    X_llm, y_llm, stats = synthesize(
        feature_names, feature_ranges, X_train, y_train, target_desc,
        n_target_rows=500,
        cache_dir=os.path.join(DATA, "llm_cache"),
        batch_size=40, sleep_s=8.0,
        max_retries=None,  # retry through quota outages; cache is append-only
    )
    print(f"done: {X_llm.shape} rows in {(time.time()-t0)/60:.1f} min",
          flush=True)
    print(f"accepted={stats['batches_accepted']} "
          f"rejected={stats['batches_rejected']} "
          f"quarantined={len(stats.get('quarantined', []))}", flush=True)
    for r in stats["rejections"][:20]:
        print("REJECTED batch", r["batch"], "-", r.get("reason"), flush=True)
    np.save(os.path.join(DATA, "X_llm.npy"), X_llm)
    np.save(os.path.join(DATA, "y_llm.npy"), y_llm)
    with open(os.path.join(DATA, "llm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
