"""FastAPI service for the shipped baseline models, topic spikes and weekly briefs.

uvicorn src.serving.app:app --reload    # docs at http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src import config
from src.brief import retrieval

BRIEF_GLOB = "weekly_brief_????-??-??.md"
DEFAULT_RISK_THRESHOLD = 0.5


class ComplaintText(BaseModel):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.PRODUCT_MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing {config.PRODUCT_MODEL_PATH}. Run: python -m src.serving.train_baseline"
        )
    app.state.product_model = joblib.load(config.PRODUCT_MODEL_PATH)
    app.state.monetary_model = joblib.load(config.MONETARY_MODEL_PATH)

    # The topics parquet is built locally (notebooks 00 -> 01 -> 06) and is not
    # committed. Without it the API still serves /classify, /risk and /brief/latest.
    app.state.topic_labels = None
    app.state.topic_spikes = None
    if config.TOPICS_PARQUET.exists():
        df = retrieval.load_topics()
        app.state.topic_labels = retrieval.build_topic_labels(df)
        app.state.topic_spikes = retrieval.find_spikes(df)

    yield


app = FastAPI(
    title="Complaint Intelligence Platform",
    description="Classify complaints, score monetary-relief risk, see emerging "
    "issues, and read the latest weekly brief.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    state = request.app.state
    return HealthResponse(
        status="ok",
        product_model_loaded=state.product_model is not None,
        monetary_model_loaded=state.monetary_model is not None,
        topic_spikes_loaded=0 if state.topic_spikes is None else len(state.topic_spikes),
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(body: ComplaintText, request: Request):
    model = request.app.state.product_model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    probabilities = model.predict_proba([body.text])[0]
    ranked = sorted(
        zip(model.classes_, probabilities, strict=True), key=lambda p: p[1], reverse=True
    )
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
def risk(body: ComplaintText, request: Request, threshold: float = DEFAULT_RISK_THRESHOLD):
    model = request.app.state.monetary_model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    probability = float(model.predict_proba([body.text])[0][1])
    return RiskResponse(
        monetary_relief_probability=round(probability, 4),
        risk_flag=probability >= threshold,
        threshold=threshold,
    )


@app.get("/topics/trends", response_model=TrendsResponse)
def topics_trends(request: Request, limit: int = 20):
    spikes = request.app.state.topic_spikes
    labels = request.app.state.topic_labels
    if spikes is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Topics not loaded: {config.TOPICS_PARQUET.relative_to(config.ROOT)} "
                "is missing. Build it with notebooks 00, 01 and 06."
            ),
        )

    spikes_out = []
    for _, row in spikes.head(limit).iterrows():
        topic_id = int(row["topic"])
        spikes_out.append(
            TopicSpike(
                topic=topic_id,
                label=labels.get(topic_id, f"topic {topic_id}"),
                week=row["week"].strftime("%Y-%m-%d"),
                count=int(row["count"]),
                z_score=float(row["z_score"]),
            )
        )
    return TrendsResponse(total_spikes=len(spikes), returned=len(spikes_out), spikes=spikes_out)


@app.get("/brief/latest")
def brief_latest():
    # File names end in the week date, so the last one sorted is the newest.
    files = sorted(config.REPORTS_DIR.glob(BRIEF_GLOB))
    if not files:
        raise HTTPException(status_code=404, detail="No weekly brief found yet.")

    latest = files[-1]
    return {"file": latest.name, "markdown": latest.read_text()}
