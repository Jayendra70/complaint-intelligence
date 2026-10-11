"""
Phase 6 - Smoke tests for the FastAPI service.

These use FastAPI's TestClient, which starts the app in-process (running the
start-up step that loads the models) and lets us call the endpoints with no real
server and no network. Run with:  pytest tests/test_api.py

They are "smoke" tests: they check each endpoint responds with a sensibly-shaped
answer, not that the predictions are correct (accuracy is measured in Phase 2/3).

The two baseline models are committed, so these run on a fresh clone. The one
test that needs the locally built topics parquet is skipped when it is missing.
"""

import os

import pytest
from fastapi.testclient import TestClient

import src.serving.app as app_module
from src.brief import retrieval
from src.serving.app import app

# One client shared by every test (start-up / model loading happens once).
client = TestClient(app)

# data/processed/complaint_topics.parquet is built by notebooks 00 -> 01 -> 06
# and is gitignored, so it only exists on a machine that ran the pipeline.
needs_topics = pytest.mark.skipif(
    not os.path.exists(retrieval.TOPICS_PARQUET),
    reason="needs data/processed/complaint_topics.parquet (run notebooks 00, 01, 06)",
)

# A realistic credit-reporting complaint used by the classify/risk tests.
SAMPLE_TEXT = (
    "I found several inaccurate accounts on my credit report from XXXX and the "
    "credit bureau has not corrected them after my dispute. This is hurting my "
    "credit score and I want these errors removed."
)

# The nine product classes the model can return.
VALID_PRODUCTS = {
    "Checking or savings account",
    "Credit card or prepaid card",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, or personal loan",
    "Student loan",
    "Vehicle loan or lease",
}


def test_health_ok():
    """The health endpoint reports the service is up with models loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["product_model_loaded"] is True
    assert body["monetary_model_loaded"] is True


def test_classify_returns_valid_product():
    """/classify returns one of the nine products and a confidence in 0..1."""
    response = client.post("/classify", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["product"] in VALID_PRODUCTS
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top3"]) == 3


def test_risk_returns_probability():
    """/risk returns a probability in 0..1 and a boolean flag."""
    response = client.post("/risk", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["monetary_relief_probability"] <= 1.0
    assert isinstance(body["risk_flag"], bool)


def test_classify_rejects_empty_text():
    """An empty narrative fails validation (HTTP 422), not a 500 crash."""
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 422


@needs_topics
def test_topics_trends_returns_spikes():
    """/topics/trends returns the Phase 4 spikes, biggest first."""
    response = client.get("/topics/trends?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["total_spikes"] == 54          # locked Phase 4 number
    assert body["returned"] == 5
    assert len(body["spikes"]) == 5
    # Sorted by z-score, high to low.
    z_scores = [s["z_score"] for s in body["spikes"]]
    assert z_scores == sorted(z_scores, reverse=True)


def test_topics_trends_503_when_topics_missing(monkeypatch):
    """Without the topics data the endpoint answers 503 instead of crashing."""
    monkeypatch.setattr(app_module, "topic_spikes", None)
    response = client.get("/topics/trends")
    assert response.status_code == 503


def test_brief_latest_returns_markdown():
    """/brief/latest returns the newest weekly brief as markdown text."""
    response = client.get("/brief/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["file"].startswith("weekly_brief_")
    assert "Headline" in body["markdown"] or "Weekly Complaint Brief" in body["markdown"]
