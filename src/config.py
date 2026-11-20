"""Paths and shared constants, resolved from the repo root so scripts run from any directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"
TOPICS_PARQUET = DATA_DIR / "complaint_topics.parquet"

MODELS_DIR = ROOT / "models"
PRODUCT_MODEL_PATH = MODELS_DIR / "baseline_product.joblib"
MONETARY_MODEL_PATH = MODELS_DIR / "baseline_monetary.joblib"
DRIFT_REFERENCE_PATH = MODELS_DIR / "drift_reference.json"

REPORTS_DIR = ROOT / "reports"

TEXT_COLUMN = "consumer_complaint_narrative"
RANDOM_STATE = 42
