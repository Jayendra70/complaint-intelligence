"""Drift monitor: quiet on the training data itself, fires on shortened narratives."""

import pandas as pd
import pytest

from src import config
from src.monitoring import drift

pytestmark = pytest.mark.skipif(
    not config.TRAIN_PARQUET.exists(),
    reason="needs data/processed/train.parquet (run notebooks 00 and 01)",
)


def test_no_drift_against_training_data():
    report = drift.run_drift_check(config.TRAIN_PARQUET)
    checks = {c["metric"]: c for c in report["checks"]}
    assert checks["narrative_length_psi"]["value"] < drift.PSI_MODERATE
    assert checks["predicted_product_psi"]["value"] < drift.PSI_MODERATE


def test_drift_flagged_on_shortened_text(tmp_path):
    shortened = pd.read_parquet(config.TRAIN_PARQUET)
    text = config.TEXT_COLUMN
    shortened[text] = shortened[text].astype(str).str.split().str[:3].str.join(" ")
    path = tmp_path / "shortened.parquet"
    shortened.to_parquet(path)

    report = drift.run_drift_check(path)
    checks = {c["metric"]: c for c in report["checks"]}
    assert checks["narrative_length_ks"]["drifted"] is True
    assert report["drift_detected"] is True
