"""Smoke tests for the FastAPI service. Accuracy is measured in the notebooks, not here."""

import pytest
from fastapi.testclient import TestClient

from src import config
from src.serving.app import app

needs_topics = pytest.mark.skipif(
    not config.TOPICS_PARQUET.exists(),
    reason="needs data/processed/complaint_topics.parquet (run notebooks 00, 01, 06)",
)

SAMPLE_TEXT = (
    "I found several inaccurate accounts on my credit report from XXXX and the "
    "credit bureau has not corrected them after my dispute. This is hurting my "
    "credit score and I want these errors removed."
)

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


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["product_model_loaded"] is True
    assert body["monetary_model_loaded"] is True


def test_classify_returns_valid_product(client):
    response = client.post("/classify", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["product"] in VALID_PRODUCTS
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top3"]) == 3


def test_risk_returns_probability(client):
    response = client.post("/risk", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["monetary_relief_probability"] <= 1.0
    assert isinstance(body["risk_flag"], bool)


def test_classify_rejects_empty_text(client):
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 422


@needs_topics
def test_topics_trends_returns_spikes(client):
    response = client.get("/topics/trends?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["total_spikes"] == 54  # reports/topics_report.md
    assert body["returned"] == 5
    assert len(body["spikes"]) == 5
    z_scores = [s["z_score"] for s in body["spikes"]]
    assert z_scores == sorted(z_scores, reverse=True)


def test_topics_trends_503_when_topics_missing(client, monkeypatch):
    monkeypatch.setattr(client.app.state, "topic_spikes", None)
    response = client.get("/topics/trends")
    assert response.status_code == 503


def test_brief_latest_returns_markdown(client):
    response = client.get("/brief/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["file"].startswith("weekly_brief_")
    assert "Headline" in body["markdown"] or "Weekly Complaint Brief" in body["markdown"]
