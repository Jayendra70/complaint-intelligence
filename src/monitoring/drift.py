"""
Phase 6 - Drift monitoring.

A model trained on old data can quietly go stale when the incoming complaints
start to look different. This script measures that "drift" by comparing a NEW
batch of complaints against the reference snapshot saved at training time
(models/drift_reference.json, written by src/serving/train_baseline.py).

It checks three things:
  1. Narrative length - PSI (Population Stability Index) on the length bins.
  2. Narrative length - KS test (a second, independent length check).
  3. Predicted product mix - PSI on how often the model predicts each product.

PSI is a standard, easy-to-read drift number:
    PSI < 0.10  -> no real drift
    0.10-0.25   -> moderate drift, keep an eye on it
    PSI > 0.25  -> major drift, the model may need retraining

It writes a small JSON + markdown report into reports/ that the Phase 7
dashboard can read.

Run it (from the repo root):
    python -m src.monitoring.drift --current data/processed/test.parquet
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp


# ----------------------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------------------

DRIFT_REFERENCE_PATH = "models/drift_reference.json"
PRODUCT_MODEL_PATH = "models/baseline_product.joblib"
TEXT_COLUMN = "consumer_complaint_narrative"

REPORT_JSON = "reports/drift_report.json"
REPORT_MD = "reports/drift_report.md"

# PSI thresholds (industry-standard rules of thumb).
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25

# Cap how many rows we predict on, so the check stays fast on big files.
MAX_PREDICT_ROWS = 20000
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. The PSI calculation
# ----------------------------------------------------------------------

def psi_from_props(ref_props, cur_props):
    """
    Population Stability Index between two sets of proportions that line up
    bucket-for-bucket. A tiny epsilon avoids divide-by-zero / log(0) on empty
    buckets.
    """
    ref = np.asarray(ref_props, dtype=float)
    cur = np.asarray(cur_props, dtype=float)
    epsilon = 1e-6
    ref = np.clip(ref, epsilon, None)
    cur = np.clip(cur, epsilon, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def severity(psi_value):
    """Turn a PSI number into a plain-English label."""
    if psi_value < PSI_MODERATE:
        return "none"
    if psi_value < PSI_MAJOR:
        return "moderate"
    return "major"


# ----------------------------------------------------------------------
# 2. The three checks
# ----------------------------------------------------------------------

def word_count(text_series):
    """Number of words in each narrative."""
    return text_series.astype(str).str.split().apply(len)


def check_length_psi(reference, current_lengths):
    """PSI on narrative length, using the reference's saved bin edges."""
    bin_edges = np.asarray(reference["length_bin_edges"], dtype=float)
    ref_props = reference["length_ref_props"]

    cur_counts, _ = np.histogram(current_lengths, bins=bin_edges)
    cur_props = cur_counts / max(cur_counts.sum(), 1)

    value = psi_from_props(ref_props, cur_props)
    return {"metric": "narrative_length_psi", "value": round(value, 4),
            "severity": severity(value)}


def check_length_ks(reference, current_lengths):
    """KS test between the reference length sample and the current lengths."""
    ref_sample = np.asarray(reference["length_sample"], dtype=float)
    result = ks_2samp(ref_sample, np.asarray(current_lengths, dtype=float))
    # KS statistic is 0 (identical) .. 1 (totally different). p<0.05 = differ.
    return {"metric": "narrative_length_ks", "statistic": round(float(result.statistic), 4),
            "p_value": round(float(result.pvalue), 6),
            "drifted": bool(result.pvalue < 0.05)}


def check_product_psi(reference, current_df, product_model):
    """PSI on the predicted-product mix vs the training-time predicted mix."""
    ref_props = reference["product_ref_props"]

    # Predict on (a sample of) the current data.
    if len(current_df) > MAX_PREDICT_ROWS:
        sample = current_df.sample(MAX_PREDICT_ROWS, random_state=RANDOM_STATE)
    else:
        sample = current_df
    predicted = product_model.predict(sample[TEXT_COLUMN])
    cur_counts = pd.Series(predicted).value_counts()
    cur_props_map = (cur_counts / cur_counts.sum()).to_dict()

    # Line the two distributions up over the same set of products.
    all_products = sorted(set(ref_props) | set(cur_props_map))
    ref_aligned = [ref_props.get(p, 0.0) for p in all_products]
    cur_aligned = [cur_props_map.get(p, 0.0) for p in all_products]

    value = psi_from_props(ref_aligned, cur_aligned)
    return {"metric": "predicted_product_psi", "value": round(value, 4),
            "severity": severity(value)}


# ----------------------------------------------------------------------
# 3. Run all checks and build a report dictionary
# ----------------------------------------------------------------------

def run_drift_check(current_parquet):
    """Load the reference + current data, run the three checks, return a dict."""
    with open(DRIFT_REFERENCE_PATH) as f:
        reference = json.load(f)

    current_df = pd.read_parquet(current_parquet)
    current_lengths = word_count(current_df[TEXT_COLUMN]).values

    product_model = joblib.load(PRODUCT_MODEL_PATH)

    length_psi = check_length_psi(reference, current_lengths)
    length_ks = check_length_ks(reference, current_lengths)
    product_psi = check_product_psi(reference, current_df, product_model)

    # Overall verdict: drift if either PSI is moderate-or-worse, or KS says so.
    drift_detected = (
        length_psi["severity"] != "none"
        or product_psi["severity"] != "none"
        or length_ks["drifted"]
    )

    return {
        "current_data": os.path.basename(current_parquet),
        "reference_rows": reference["n_reference_rows"],
        "current_rows": int(len(current_df)),
        "checks": [length_psi, length_ks, product_psi],
        "drift_detected": bool(drift_detected),
    }


# ----------------------------------------------------------------------
# 4. Save the report (JSON + markdown for the dashboard)
# ----------------------------------------------------------------------

def save_report(report):
    os.makedirs("reports", exist_ok=True)

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Drift report", ""]
    lines.append(f"- Current data: `{report['current_data']}` "
                 f"({report['current_rows']} rows)")
    lines.append(f"- Reference rows (training time): {report['reference_rows']}")
    lines.append(f"- **Drift detected: {report['drift_detected']}**")
    lines.append("")
    lines.append("| Check | Value | Verdict |")
    lines.append("|---|---|---|")
    for check in report["checks"]:
        if check["metric"] == "narrative_length_ks":
            value = f"KS={check['statistic']} (p={check['p_value']})"
            verdict = "drifted" if check["drifted"] else "ok"
        else:
            value = f"PSI={check['value']}"
            verdict = check["severity"]
        lines.append(f"| {check['metric']} | {value} | {verdict} |")
    lines.append("")
    lines.append("PSI guide: <0.10 none, 0.10-0.25 moderate, >0.25 major. "
                 "KS: p<0.05 means the length distributions differ.")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# 5. Command line entry point
# ----------------------------------------------------------------------

def _get_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    current = _get_arg("--current", "data/processed/test.parquet")
    print(f"Comparing {current} against the training reference ...")
    report = run_drift_check(current)
    save_report(report)

    print(f"\nDrift detected: {report['drift_detected']}")
    for check in report["checks"]:
        print(" ", check)
    print(f"\nSaved {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
