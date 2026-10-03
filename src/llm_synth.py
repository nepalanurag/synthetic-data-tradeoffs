"""LLM row synthesis via the Gemini API.

Each request asks for a batch of CSV rows following the real schema, with a
few real example rows in context. Batches are parsed defensively: wrong
column count, non-numeric cells, or out-of-range values are rejected and the
batch is retried (rejections are counted and reported honestly). Raw model
responses are cached to data/llm_cache/ so the run is reproducible without
re-calling the API.

Auth uses the stored connector credential through the skill-creator helper;
no raw key ever appears in code or logs. Only
generativelanguage.googleapis.com is contacted.
"""

import csv
import io
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import (
    add_surrogate_to_request,
    read_json_response,
)

HOSTS = ["generativelanguage.googleapis.com"]
BASE = "https://generativelanguage.googleapis.com/v1beta"
CREDENTIAL = "custom.google-gemini"
MODEL = "gemini-2.5-flash"

SEED = 20261002


def _post(path, payload, timeout=120):
    url = BASE + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    add_surrogate_to_request(req, CREDENTIAL, allowed_hosts=HOSTS)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return read_json_response(resp)


def generate_text(prompt, max_retries=6):
    """Generate one completion with exponential backoff on transient errors.

    max_retries=None retries indefinitely (for quota outages); 429s wait
    60s * (attempt+1), capped at 600s.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 8192},
    }
    last_err = None
    attempt = 0
    while max_retries is None or attempt < max_retries:
        try:
            out = _post(f"/models/{MODEL}:generateContent", payload)
            cands = out.get("candidates", [])
            if not cands:
                raise RuntimeError("no candidates in response")
            parts = cands[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        except Exception as e:  # noqa: BLE001 - retry transient API errors
            last_err = e
            if "429" in str(e):
                # Rate limit: wait longer before retrying, cap at 10 min.
                time.sleep(min(60 * (attempt + 1), 600))
            else:
                time.sleep(2**attempt * 2)
            attempt += 1
    raise RuntimeError(f"Gemini call failed after {max_retries} tries: {last_err}")


def build_prompt(feature_names, feature_ranges, example_rows, target_desc, batch_size):
    header = ",".join(["target"] + list(feature_names))
    lines = [
        "You are generating synthetic rows for a medical tabular dataset used to",
        "study synthetic-data methods. The target column meaning: " + target_desc,
        "",
        "Column ranges observed in the real training data (stay within them):",
    ]
    for name in feature_names:
        lo, hi = feature_ranges[name]
        lines.append(f"  {name}: numeric, observed range [{lo:.4g}, {hi:.4g}]")
    lines.append("")
    lines.append(
        "A few real example rows (header: " + header + "). "
        "Generate NEW rows that are plausible for this distribution "
        "but not copies of these examples:"
    )
    lines.append(header)
    for row in example_rows:
        lines.append(",".join(f"{v:.4g}" for v in row))
    lines.append("")
    lines.append(
        f"Now output exactly {batch_size} new rows as plain CSV with the same "
        "header first, no markdown fences, no commentary, no extra columns."
    )
    return "\n".join(lines)


def parse_batch(text, feature_names, feature_ranges, slack=0.02):
    """Parse one CSV batch. Returns (rows, info) where rows is None if rejected.

    Tolerant of model chatter: finds the header line by matching the expected
    column names, then reads data rows until the first line that does not
    parse as a full numeric row (trailing prose is ignored and counted).
    info always carries a machine-readable account of what happened.
    """
    info = {"skipped_prose": 0}
    expected = ["target"] + [n.strip().lower() for n in feature_names]
    text = text.strip().replace("```csv", "").replace("```", "")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and "," in ln]
    header_idx = None
    for i, ln in enumerate(lines):
        cells = [c.strip().lower() for c in ln.split(",")]
        if cells == expected:
            header_idx = i
            break
    if header_idx is None:
        return None, {"reason": "no header line matching expected columns",
                      "skipped_prose": len(lines)}
    rows = []
    for ln in lines[header_idx + 1:]:
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) != len(expected):
            info["skipped_prose"] += 1
            continue
        try:
            vals = [float(c) for c in cells]
        except ValueError:
            info["skipped_prose"] += 1
            continue
        if not all(abs(v) < 1e12 for v in vals):
            return None, {"reason": f"absurd magnitude in line: {ln[:60]}",
                          **info}
        if vals[0] not in (0.0, 1.0):
            return None, {"reason": f"target not 0/1 in line: {ln[:60]}",
                          **info}
        bad = None
        for name, v in zip(feature_names, vals[1:]):
            lo, hi = feature_ranges[name]
            span = hi - lo
            if v < lo - slack * span or v > hi + slack * span:
                bad = f"{name}={v:.4g} out of range [{lo:.4g}, {hi:.4g}]"
                break
        if bad:
            return None, {"reason": bad, **info}
        rows.append(vals)
    if not rows:
        return None, {"reason": "no valid data rows after header", **info}
    info["n_rows"] = len(rows)
    return rows, info


def synthesize(
    feature_names,
    feature_ranges,
    X_train,
    y_train,
    target_desc,
    n_target_rows,
    cache_dir,
    batch_size=20,
    sleep_s=1.0,
    max_retries=None,
):
    """Generate n_target_rows validated synthetic rows. Returns (rows, stats)."""
    import numpy as np

    os.makedirs(cache_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(X_train))
    ex_idx = order[:8]
    example_rows = [
        [float(y_train[i])] + [float(v) for v in X_train[i]] for i in ex_idx
    ]
    prompt = build_prompt(
        feature_names, feature_ranges, example_rows, target_desc, batch_size
    )
    rows = []
    stats = {"batches_accepted": 0, "batches_rejected": 0, "rejections": [],
             "quarantined": []}
    # Reuse any previously cached batches (append-only cache: files are never
    # deleted; unparseable ones are renamed to .quarantined and logged).
    attempt_no = 0
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".txt") or fname.endswith(".quarantined.txt"):
            continue
        with open(os.path.join(cache_dir, fname)) as f:
            text = f.read()
        parsed, info = parse_batch(text, feature_names, feature_ranges)
        if parsed is None:
            qname = fname.replace(".txt", ".quarantined.txt")
            os.rename(os.path.join(cache_dir, fname),
                      os.path.join(cache_dir, qname))
            stats["quarantined"].append({"file": fname, **info})
            continue
        stats["batches_accepted"] += 1
        rows.extend(parsed)
        attempt_no += 1
    while len(rows) < n_target_rows:
        attempt_no += 1
        cache_path = os.path.join(cache_dir, f"batch_{attempt_no:03d}.txt")
        if os.path.exists(cache_path):
            # Already counted above; skip (should not happen).
            continue
        text = generate_text(prompt, max_retries=max_retries)
        with open(cache_path, "w") as f:
            f.write(text)
        parsed, info = parse_batch(text, feature_names, feature_ranges)
        if parsed is None:
            stats["batches_rejected"] += 1
            stats["rejections"].append({"batch": attempt_no, **info})
            time.sleep(sleep_s)
            continue
        stats["batches_accepted"] += 1
        rows.extend(parsed)
        time.sleep(sleep_s)
    rows = rows[:n_target_rows]
    arr = np.array(rows)
    y_syn = arr[:, 0].astype(int)
    X_syn = arr[:, 1:]
    return X_syn, y_syn, stats
