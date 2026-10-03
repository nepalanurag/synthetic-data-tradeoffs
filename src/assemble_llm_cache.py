"""Assemble LLM synthetic rows from the existing cache, no new API calls.

Mirrors the cache-reuse branch of llm_synth.synthesize(): each cached
batch file is parsed with parse_batch(); unparseable files are renamed to
.quarantined.txt exactly as synthesize() does. Writes data/X_llm.npy,
data/y_llm.npy, data/llm_stats.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_synth import parse_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "llm_cache")


def main():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    feature_names = [c for c in train.columns if c != "target"]
    X_train = train[feature_names].values
    feature_ranges = {
        n: (float(X_train[:, j].min()), float(X_train[:, j].max()))
        for j, n in enumerate(feature_names)
    }
    rows = []
    stats = {"batches_accepted": 0, "batches_rejected": 0,
             "rejections": [], "quarantined": []}
    for fname in sorted(os.listdir(CACHE)):
        if not fname.endswith(".txt") or fname.endswith(".quarantined.txt"):
            continue
        with open(os.path.join(CACHE, fname)) as f:
            text = f.read()
        parsed, info = parse_batch(text, feature_names, feature_ranges)
        if parsed is None:
            qname = fname.replace(".txt", ".quarantined.txt")
            os.rename(os.path.join(CACHE, fname),
                      os.path.join(CACHE, qname))
            stats["batches_rejected"] += 1
            stats["rejections"].append({"batch": fname, **info})
            stats["quarantined"].append({"file": fname, **info})
            continue
        stats["batches_accepted"] += 1
        rows.extend(parsed)
    arr = np.array(rows)
    X_syn = arr[:, 1:]
    y_syn = arr[:, 0].astype(int)
    np.save(os.path.join(DATA, "X_llm.npy"), X_syn)
    np.save(os.path.join(DATA, "y_llm.npy"), y_syn)
    with open(os.path.join(DATA, "llm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"rows={len(rows)} accepted={stats['batches_accepted']} "
          f"rejected={stats['batches_rejected']}")
    print("class balance:", np.bincount(y_syn).tolist())


if __name__ == "__main__":
    main()
