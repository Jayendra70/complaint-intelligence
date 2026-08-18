"""
Phase 6 - Train and SAVE the shipped baseline models so the API can load them.

Phase 2 trained the baseline inside notebook 03 but never saved it to disk. The
FastAPI service needs the fitted model on disk, so this script re-fits the exact
same baseline (same features, same settings as notebook 03) and saves it.

It fits two models, both text-only:
  1. product  - 9-class LogisticRegression  (which product the complaint is about)
  2. monetary - binary   LogisticRegression  (will the company pay money back?)

It also, as a check, scores them on the held-out test set and prints the numbers,
which must match the locked Phase 2 baseline (product macro-F1 0.7065,
monetary PR-AUC 0.3337).

Finally it writes models/drift_reference.json - the "what normal looks like"
snapshot the drift monitor (src/monitoring/drift.py) compares new data against.

Run it once (on the Mac, after recreating the venv):
    python -m src.serving.train_baseline
"""

import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score


# ----------------------------------------------------------------------
# 0. Settings (all paths and knobs in one place)
# ----------------------------------------------------------------------

TRAIN_PARQUET = "data/processed/train.parquet"
TEST_PARQUET = "data/processed/test.parquet"

MODELS_DIR = "models"
PRODUCT_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_product.joblib")
MONETARY_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_monetary.joblib")
DRIFT_REFERENCE_PATH = os.path.join(MODELS_DIR, "drift_reference.json")

TEXT_COLUMN = "consumer_complaint_narrative"
PRODUCT_LABEL = "product"
MONETARY_LABEL = "is_monetary_relief"

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Build the feature+model pipeline (exactly the Phase 2 baseline)
# ----------------------------------------------------------------------

def build_pipeline():
    """
    One pipeline = TF-IDF word features AND char features side by side, fed to
    a class-balanced Logistic Regression. Same settings as notebook 03, so the
    scores come out the same. Saving the whole pipeline means the API just calls
    .predict() on raw text - the text-to-numbers step travels with the model.
    """
    # Word features: single words and 2-word phrases seen in >= 3 complaints.
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True
    )
    # Character features: 3-to-5-char chunks (robust to typos and XXXX masks).
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3,
        max_features=60000, sublinear_tf=True
    )
    # Both feature sets glued together, word first then char (same as notebook 03).
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

    classifier = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
    )

    return Pipeline([("features", features), ("clf", classifier)])


# ----------------------------------------------------------------------
# 2. Build the drift reference snapshot
# ----------------------------------------------------------------------

def word_count(text_series):
    """Number of words in each narrative (a plain, fast length measure)."""
    return text_series.astype(str).str.split().apply(len)


def build_drift_reference(train, product_pipeline):
    """
    Save a small snapshot of 'normal' so the monitor can spot drift later:
      - length_bin_edges / length_ref_props : the shape of narrative lengths
      - length_sample                       : 5000 raw lengths (for the KS test)
      - product_ref_props                   : how often the model predicts each product
    """
    # 2a. Narrative-length distribution on the training data.
    lengths = word_count(train[TEXT_COLUMN])
    # 10 bins by quantile so each bin holds a similar share of training rows.
    bin_edges = np.quantile(lengths, np.linspace(0, 1, 11))
    bin_edges = np.unique(bin_edges)  # drop any duplicate edges
    ref_counts, _ = np.histogram(lengths, bins=bin_edges)
    ref_props = ref_counts / ref_counts.sum()

    # A random sample of raw lengths, used by the KS test in the monitor.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(5000, len(lengths))
    length_sample = rng.choice(lengths.values, size=sample_size, replace=False)

    # 2b. Predicted-product distribution on the training data.
    predicted = product_pipeline.predict(train[TEXT_COLUMN])
    product_counts = pd.Series(predicted).value_counts()
    product_props = (product_counts / product_counts.sum()).to_dict()

    reference = {
        "length_bin_edges": [float(x) for x in bin_edges],
        "length_ref_props": [float(x) for x in ref_props],
        "length_sample": [int(x) for x in length_sample],
        "product_ref_props": {str(k): float(v) for k, v in product_props.items()},
        "n_reference_rows": int(len(train)),
    }
    with open(DRIFT_REFERENCE_PATH, "w") as f:
        json.dump(reference, f, indent=2)
    print("Saved", DRIFT_REFERENCE_PATH)


# ----------------------------------------------------------------------
# 3. Main: fit, score, save
# ----------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data ...")
    train = pd.read_parquet(TRAIN_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    print("  train:", len(train), " test:", len(test))

    # --- Product model -------------------------------------------------
    print("\nFitting product model (9-class) ...")
    product_pipeline = build_pipeline()
    product_pipeline.fit(train[TEXT_COLUMN], train[PRODUCT_LABEL])

    product_pred = product_pipeline.predict(test[TEXT_COLUMN])
    product_macro_f1 = f1_score(test[PRODUCT_LABEL], product_pred, average="macro")
    print("  test macro-F1:", round(product_macro_f1, 4), " (Phase 2 locked: 0.7065)")

    joblib.dump(product_pipeline, PRODUCT_MODEL_PATH)
    print("  saved", PRODUCT_MODEL_PATH)

    # --- Monetary model ------------------------------------------------
    print("\nFitting monetary-relief model (binary) ...")
    monetary_pipeline = build_pipeline()
    monetary_pipeline.fit(train[TEXT_COLUMN], train[MONETARY_LABEL])

    monetary_scores = monetary_pipeline.predict_proba(test[TEXT_COLUMN])[:, 1]
    monetary_pr_auc = average_precision_score(test[MONETARY_LABEL], monetary_scores)
    print("  test PR-AUC:", round(monetary_pr_auc, 4), " (Phase 2 locked: 0.3337)")

    joblib.dump(monetary_pipeline, MONETARY_MODEL_PATH)
    print("  saved", MONETARY_MODEL_PATH)

    # --- Drift reference ----------------------------------------------
    print("\nBuilding drift reference snapshot ...")
    build_drift_reference(train, product_pipeline)

    print("\nDone. Models and drift reference are in the models/ folder.")


if __name__ == "__main__":
    main()
