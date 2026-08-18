"""
Phase 6 - Smoke test for the drift monitor.

Two quick checks:
  1. Comparing the training data against itself shows (almost) no drift - the
     metrics are near zero. This proves the monitor doesn't cry wolf.
  2. Comparing against a deliberately shortened batch pushes the length KS test
     to flag drift - proving the monitor actually reacts when the data changes.

Run with:  pytest tests/test_drift.py
"""

import pandas as pd

from src.monitoring import drift


def test_no_drift_against_training_data():
    """Train-vs-train: PSI numbers should be tiny (no real drift)."""
    report = drift.run_drift_check("data/processed/train.parquet")
    checks = {c["metric"]: c for c in report["checks"]}
    # Both PSIs are essentially zero when comparing the reference to itself.
    assert checks["narrative_length_psi"]["value"] < drift.PSI_MODERATE
    assert checks["predicted_product_psi"]["value"] < drift.PSI_MODERATE


def test_drift_flagged_on_shortened_text(tmp_path):
    """Chop every narrative down to a few words -> the length KS test flags it."""
    train = pd.read_parquet("data/processed/train.parquet")
    shortened = train.copy()
    shortened[drift.TEXT_COLUMN] = (
        shortened[drift.TEXT_COLUMN].astype(str).str.split().str[:3].str.join(" ")
    )
    path = tmp_path / "shortened.parquet"
    shortened.to_parquet(path)

    report = drift.run_drift_check(str(path))
    checks = {c["metric"]: c for c in report["checks"]}
    assert checks["narrative_length_ks"]["drifted"] is True
    assert report["drift_detected"] is True
