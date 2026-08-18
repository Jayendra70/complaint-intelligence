"""
Phase 6 - FastAPI service for the Complaint Intelligence Platform.

It turns the pieces built in earlier phases into a running web service with four
useful endpoints (plus a health check):

  GET  /health          - is the service up and are the models loaded?
  POST /classify        - given a complaint's text, predict its product (9 classes)
  POST /risk            - given a complaint's text, score its monetary-relief risk
  GET  /topics/trends   - the emerging-issue spikes found in Phase 4
  GET  /brief/latest     - the most recent weekly brief from Phase 5

Design notes (kept deliberately simple):
  - The two baseline models are loaded ONCE when the service starts, not on every
    request, so requests are fast.
  - The topic spikes are computed once at start-up too (reusing the Phase 5
    retrieval code), because the underlying data does not change while running.
  - Request and response shapes are declared with pydantic, so FastAPI validates
    inputs automatically and documents everything at /docs.

Run it (from the repo root):
    uvicorn src.serving.app:app --reload
Then open http://localhost:8000/docs
"""

import glob
import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.brief import retrieval


# ----------------------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------------------

PRODUCT_MODEL_PATH = "models/baseline_product.joblib"
MONETARY_MODEL_PATH = "models/baseline_monetary.joblib"
BRIEF_GLOB = "reports/weekly_brief_*.md"

# Default cut-off for the risk flag. Above this predicted probability we flag a
# complaint as "likely to end in monetary relief" for a human to look at.
DEFAULT_RISK_THRESHOLD = 0.5


# ----------------------------------------------------------------------
# 1. Load everything once at start-up
# ----------------------------------------------------------------------
# These are filled by load_artifacts(). We keep them at module level so every
# request reuses the same in-memory models instead of reloading from disk.

product_model = None
monetary_model = None
topic_labels = None
topic_spikes = None


def load_artifacts():
    """Load the models and compute the topic spikes. Called once at start-up."""
    global product_model, monetary_model, topic_labels, topic_spikes

    # 1a. The two saved baseline models (built by src/serving/train_baseline.py).
    if not os.path.exists(PRODUCT_MODEL_PATH):
        raise RuntimeError(
            f"Missing {PRODUCT_MODEL_PATH}. Run: python -m src.serving.train_baseline"
        )
    product_model = joblib.load(PRODUCT_MODEL_PATH)
    monetary_model = joblib.load(MONETARY_MODEL_PATH)

    # 1b. Topic spikes for the /topics/trends endpoint (Phase 5 retrieval code).
    df = retrieval.load_topics()
    topic_labels = retrieval.build_topic_labels(df)
    topic_spikes = retrieval.find_spikes(df)


# ----------------------------------------------------------------------
# 2. Request / response shapes (pydantic models)
# ----------------------------------------------------------------------

class ComplaintText(BaseModel):
    """The one input both /classify and /risk take: a complaint's narrative."""
    text: str = Field(min_length=1, description="The complaint narrative text.")


class ProductProbability(BaseModel):
    product: str
    probability: float


class ClassifyResponse(BaseModel):
    product: str
    confidence: float
    top3: list[ProductProbability]


class RiskResponse(BaseModel):
    monetary_relief_probability: float
    risk_flag: bool
    threshold: float


class TopicSpike(BaseModel):
    topic: int
    label: str
    week: str
    count: int
    z_score: float


class TrendsResponse(BaseModel):
    total_spikes: int
    returned: int
    spikes: list[TopicSpike]


class HealthResponse(BaseModel):
    status: str
    product_model_loaded: bool
    monetary_model_loaded: bool
    topic_spikes_loaded: int


# ----------------------------------------------------------------------
# 3. The app
# ----------------------------------------------------------------------

app = FastAPI(
    title="Complaint Intelligence Platform",
    description="Classify complaints, score monetary-relief risk, see emerging "
                "issues, and read the latest weekly brief.",
    version="1.0.0",
)

# Load the models and topic spikes as soon as the app module is imported. This
# runs once - whether started by uvicorn, by Docker, or by the test client - so
# every request afterwards reuses the in-memory models.
load_artifacts()


# ----------------------------------------------------------------------
# 4. Endpoints
# ----------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Simple liveness check the dashboard and Docker can poll."""
    return HealthResponse(
        status="ok",
        product_model_loaded=product_model is not None,
        monetary_model_loaded=monetary_model is not None,
        topic_spikes_loaded=0 if topic_spikes is None else len(topic_spikes),
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(body: ComplaintText):
    """Predict which product a complaint is about, with the top-3 probabilities."""
    if product_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # predict_proba returns one probability per class, in model.classes_ order.
    probabilities = product_model.predict_proba([body.text])[0]
    classes = product_model.classes_

    # Sort classes by probability, high to low, and keep the top 3.
    ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
    top3 = [
        ProductProbability(product=str(name), probability=round(float(prob), 4))
        for name, prob in ranked[:3]
    ]

    return ClassifyResponse(
        product=str(ranked[0][0]),
        confidence=round(float(ranked[0][1]), 4),
        top3=top3,
    )


@app.post("/risk", response_model=RiskResponse)
def risk(body: ComplaintText, threshold: float = DEFAULT_RISK_THRESHOLD):
    """Score how likely a complaint is to end in monetary relief (0..1)."""
    if monetary_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    probability = float(monetary_model.predict_proba([body.text])[0][1])
    return RiskResponse(
        monetary_relief_probability=round(probability, 4),
        risk_flag=probability >= threshold,
        threshold=threshold,
    )


@app.get("/topics/trends", response_model=TrendsResponse)
def topics_trends(limit: int = 20):
    """Return the emerging-issue spikes from Phase 4, biggest z-score first."""
    if topic_spikes is None:
        raise HTTPException(status_code=503, detail="Topics not loaded.")

    spikes_out = []
    for _, row in topic_spikes.head(limit).iterrows():
        topic_id = int(row["topic"])
        spikes_out.append(
            TopicSpike(
                topic=topic_id,
                label=topic_labels.get(topic_id, f"topic {topic_id}"),
                week=row["week"].strftime("%Y-%m-%d"),
                count=int(row["count"]),
                z_score=float(row["z_score"]),
            )
        )

    return TrendsResponse(
        total_spikes=len(topic_spikes),
        returned=len(spikes_out),
        spikes=spikes_out,
    )


@app.get("/brief/latest")
def brief_latest():
    """Return the most recent weekly brief (raw markdown text)."""
    files = sorted(glob.glob(BRIEF_GLOB))
    if not files:
        raise HTTPException(status_code=404, detail="No weekly brief found yet.")

    # File names end in the week date (weekly_brief_YYYY-MM-DD.md), so the last
    # one alphabetically is the most recent week.
    latest_path = files[-1]
    with open(latest_path) as f:
        text = f.read()

    return {"file": os.path.basename(latest_path), "markdown": text}
