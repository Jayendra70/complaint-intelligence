"""Compare a batch of complaints against the training-time reference for drift.

Checks narrative-length PSI, a narrative-length KS test, and PSI on the predicted
product mix. Writes reports/drift_report.json and reports/drift_report.md.

    python -m src.monitoring.drift --current data/processed/test.parquet
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import config

REPORT_JSON = config.REPORTS_DIR / "drift_report.json"
REPORT_MD = config.REPORTS_DIR / "drift_report.md"

# Common PSI rule of thumb: < 0.10 none, 0.10-0.25 moderate, > 0.25 major.
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25

MAX_PREDICT_ROWS = 20000


def word_count(text_series: pd.Series) -> pd.Series:
    return text_series.astype(str).str.split().apply(len)


def psi_from_props(ref_props: Sequence[float], cur_props: Sequence[float]) -> float:
    """Population Stability Index between two aligned sets of bucket proportions."""
    # Clip so empty buckets don't produce log(0) or divide by zero.
    epsilon = 1e-6
    ref = np.clip(np.asarray(ref_props, dtype=float), epsilon, None)
    cur = np.clip(np.asarray(cur_props, dtype=float), epsilon, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def severity(psi_value: float) -> str:
    if psi_value < PSI_MODERATE:
        return "none"
    if psi_value < PSI_MAJOR:
        return "moderate"
    return "major"


def check_length_psi(reference: Mapping, current_lengths: np.ndarray) -> dict:
    bin_edges = np.asarray(reference["length_bin_edges"], dtype=float)
    cur_counts, _ = np.histogram(current_lengths, bins=bin_edges)
    cur_props = cur_counts / max(cur_counts.sum(), 1)
    value = psi_from_props(reference["length_ref_props"], cur_props)
    return {"metric": "narrative_length_psi", "value": round(value, 4), "severity": severity(value)}


def check_length_ks(reference: Mapping, current_lengths: np.ndarray) -> dict:
    ref_sample = np.asarray(reference["length_sample"], dtype=float)
    result = ks_2samp(ref_sample, np.asarray(current_lengths, dtype=float))
    return {
        "metric": "narrative_length_ks",
        "statistic": round(float(result.statistic), 4),
        "p_value": round(float(result.pvalue), 6),
        "drifted": bool(result.pvalue < 0.05),
    }


def check_product_psi(reference: Mapping, current_df: pd.DataFrame, product_model) -> dict:
    ref_props = reference["product_ref_props"]

    if len(current_df) > MAX_PREDICT_ROWS:
        sample = current_df.sample(MAX_PREDICT_ROWS, random_state=config.RANDOM_STATE)
    else:
        sample = current_df
    predicted = product_model.predict(sample[config.TEXT_COLUMN])
    cur_counts = pd.Series(predicted).value_counts()
    cur_props = (cur_counts / cur_counts.sum()).to_dict()

    products = sorted(set(ref_props) | set(cur_props))
    value = psi_from_props(
        [ref_props.get(p, 0.0) for p in products],
        [cur_props.get(p, 0.0) for p in products],
    )
    return {
        "metric": "predicted_product_psi",
        "value": round(value, 4),
        "severity": severity(value),
    }


def run_drift_check(current_parquet: str | Path) -> dict:
    reference = json.loads(config.DRIFT_REFERENCE_PATH.read_text())
    current_df = pd.read_parquet(current_parquet)
    current_lengths = word_count(current_df[config.TEXT_COLUMN]).values
    product_model = joblib.load(config.PRODUCT_MODEL_PATH)

    length_psi = check_length_psi(reference, current_lengths)
    length_ks = check_length_ks(reference, current_lengths)
    product_psi = check_product_psi(reference, current_df, product_model)

    drift_detected = (
        length_psi["severity"] != "none"
        or product_psi["severity"] != "none"
        or length_ks["drifted"]
    )
    return {
        "current_data": Path(current_parquet).name,
        "reference_rows": reference["n_reference_rows"],
        "current_rows": int(len(current_df)),
        "checks": [length_psi, length_ks, product_psi],
        "drift_detected": bool(drift_detected),
    }


def save_report(report: dict) -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2))

    lines = [
        "# Drift report",
        "",
        f"- Current data: `{report['current_data']}` ({report['current_rows']} rows)",
        f"- Reference rows (training time): {report['reference_rows']}",
        f"- **Drift detected: {report['drift_detected']}**",
        "",
        "| Check | Value | Verdict |",
        "|---|---|---|",
    ]
    for check in report["checks"]:
        if check["metric"] == "narrative_length_ks":
            value = f"KS={check['statistic']} (p={check['p_value']})"
            verdict = "drifted" if check["drifted"] else "ok"
        else:
            value = f"PSI={check['value']}"
            verdict = check["severity"]
        lines.append(f"| {check['metric']} | {value} | {verdict} |")
    lines += [
        "",
        "PSI guide: <0.10 none, 0.10-0.25 moderate, >0.25 major. "
        "KS: p<0.05 means the length distributions differ.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a batch of complaints for drift.")
    parser.add_argument(
        "--current",
        default=str(config.TEST_PARQUET),
        help="parquet file to compare against the training reference",
    )
    args = parser.parse_args()

    print(f"Comparing {args.current} against the training reference ...")
    report = run_drift_check(args.current)
    save_report(report)

    print(f"\nDrift detected: {report['drift_detected']}")
    for check in report["checks"]:
        print(" ", check)
    print(f"\nSaved {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
