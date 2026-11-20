"""Fit and save the shipped TF-IDF + LogisticRegression models and the drift reference.

Re-fits the notebook 03 baseline for both tasks, checks the test scores against the
reported numbers (product macro-F1 0.7065, monetary PR-AUC 0.3337), and writes
models/baseline_*.joblib and models/drift_reference.json.

    python -m src.serving.train_baseline
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config
from src.monitoring.drift import word_count

PRODUCT_LABEL = "product"
MONETARY_LABEL = "is_monetary_relief"


def build_pipeline() -> Pipeline:
    """Word 1-2-gram + char 3-5-gram TF-IDF into a class-balanced LogisticRegression."""
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000, sublinear_tf=True
    )
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
    classifier = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=config.RANDOM_STATE
    )
    return Pipeline([("features", features), ("clf", classifier)])


def build_drift_reference(train: pd.DataFrame, product_pipeline: Pipeline) -> None:
    """Save the training-time length distribution and predicted product mix."""
    lengths = word_count(train[config.TEXT_COLUMN])
    # Quantile bins so each bin holds a similar share of training rows.
    bin_edges = np.unique(np.quantile(lengths, np.linspace(0, 1, 11)))
    ref_counts, _ = np.histogram(lengths, bins=bin_edges)
    ref_props = ref_counts / ref_counts.sum()

    rng = np.random.default_rng(config.RANDOM_STATE)
    length_sample = rng.choice(lengths.values, size=min(5000, len(lengths)), replace=False)

    predicted = product_pipeline.predict(train[config.TEXT_COLUMN])
    product_counts = pd.Series(predicted).value_counts()
    product_props = (product_counts / product_counts.sum()).to_dict()

    reference = {
        "length_bin_edges": [float(x) for x in bin_edges],
        "length_ref_props": [float(x) for x in ref_props],
        "length_sample": [int(x) for x in length_sample],
        "product_ref_props": {str(k): float(v) for k, v in product_props.items()},
        "n_reference_rows": int(len(train)),
    }
    config.DRIFT_REFERENCE_PATH.write_text(json.dumps(reference, indent=2))
    print("Saved", config.DRIFT_REFERENCE_PATH)


def main() -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)

    train = pd.read_parquet(config.TRAIN_PARQUET)
    test = pd.read_parquet(config.TEST_PARQUET)
    print(f"train: {len(train)}  test: {len(test)}")
    text = config.TEXT_COLUMN

    print("Fitting product model ...")
    product_pipeline = build_pipeline()
    product_pipeline.fit(train[text], train[PRODUCT_LABEL])
    product_pred = product_pipeline.predict(test[text])
    macro_f1 = f1_score(test[PRODUCT_LABEL], product_pred, average="macro")
    print(f"  test macro-F1: {macro_f1:.4f}  (reported: 0.7065)")
    joblib.dump(product_pipeline, config.PRODUCT_MODEL_PATH)

    print("Fitting monetary-relief model ...")
    monetary_pipeline = build_pipeline()
    monetary_pipeline.fit(train[text], train[MONETARY_LABEL])
    scores = monetary_pipeline.predict_proba(test[text])[:, 1]
    pr_auc = average_precision_score(test[MONETARY_LABEL], scores)
    print(f"  test PR-AUC: {pr_auc:.4f}  (reported: 0.3337)")
    joblib.dump(monetary_pipeline, config.MONETARY_MODEL_PATH)

    build_drift_reference(train, product_pipeline)


if __name__ == "__main__":
    main()
