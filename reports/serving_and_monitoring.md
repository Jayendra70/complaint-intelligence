# Serving and monitoring

Code: `src/serving/`, `src/monitoring/`, `tests/`, `Dockerfile`, `docker-compose.yml`.

## What's built

1. **`src/serving/train_baseline.py`.** Notebook 03 trained the baseline but
   didn't save it. This script re-fits the same model (word + char TF-IDF →
   class-balanced LogisticRegression, same settings) for both tasks and saves it
   to `models/`. As a check it scores the held-out test set:

   | Task | This script | Notebook 03 |
   |---|---|---|
   | Product (macro-F1) | 0.706 | 0.7065 |
   | Monetary (PR-AUC) | 0.3339 | 0.3337 |

   The small differences come from library minor versions (pandas 3.0 vs 2.3).
   Artifacts: `models/baseline_product.joblib` (13 MB),
   `models/baseline_monetary.joblib` (5 MB), `models/drift_reference.json`.

2. **`src/serving/app.py`.** FastAPI service; models load once at start-up.
   - `GET  /health`: liveness, and which artifacts loaded.
   - `POST /classify`: narrative → predicted product, confidence and top 3.
   - `POST /risk`: narrative → monetary-relief probability, plus a flag at an adjustable threshold.
   - `GET  /topics/trends`: the 54 topic spikes, largest z-score first, with labels.
   - `GET  /brief/latest`: the newest weekly brief as markdown.

   Pydantic schemas validate requests (empty text → HTTP 422), and everything
   is documented at `/docs`.

3. **`src/monitoring/drift.py`.** Compares a new batch of complaints with the
   training-time reference. It reports narrative-length PSI, a narrative-length
   KS test and predicted-product-mix PSI, and writes `reports/drift_report.json`
   and `reports/drift_report.md`. PSI bands: < 0.10 none, 0.10–0.25 moderate,
   > 0.25 major.

4. **Tests.**
   - `tests/test_api.py`: health; classify returns a valid product; risk returns
     a 0..1 probability; empty text is rejected; trends returns the 54 spikes
     sorted; the 503 path; brief returns markdown.
   - `tests/test_drift.py`: train-vs-train shows no drift, and shortened text
     triggers the flag.

5. **Docker.** A lean image built from `requirements-serving.txt`, the
   scikit-learn stack only, no torch. `docker compose up --build` serves the API
   on `localhost:8000`.

## What the drift run showed

`python -m src.monitoring.drift --current data/processed/test.parquet` (the newer,
held-out complaints):

- **Narrative-length PSI = 0.012 (none).** Complaint length is stable over time.
- **Narrative-length KS: statistic 0.034, p ≈ 9e-5, flagged.** With 34k rows
  the KS test reacts to trivially small shifts. PSI is the number to go by here;
  KS is a secondary tripwire.
- **Predicted-product PSI = 0.79 (major).** This is expected. The reference
  comes from the class-balanced training set, while the test set has the natural
  mix (65% credit reporting), so the mix really does differ. In production the
  reference should be the natural traffic mix.

A drift monitor is only as useful as its reference distribution.

## How to run

```
pip install -r requirements.txt          # or requirements-serving.txt for the API only
python -m src.serving.train_baseline     # writes models/*.joblib + drift_reference.json (~8 min)
uvicorn src.serving.app:app --reload     # API at http://localhost:8000/docs
python -m src.monitoring.drift --current data/processed/test.parquet
pytest
docker compose up --build                # same API in a container
```

## Known limitations

- `/classify` confidence can read 1.00 on clear-cut complaints. Balanced
  LogReg is over-confident and not calibrated, so use the value for routing,
  not as a probability.
- The models are pickled with scikit-learn 1.9.0, and the serving image pins
  that version so the pickle loads. Regenerate the models if you change the
  scikit-learn version.
- `/topics/trends` needs `data/processed/complaint_topics.parquet`, which is
  built locally and not committed. Mount it into the container to serve that
  endpoint.

## Production hardening not built yet

The service detects drift; the automation around it is not built:

1. **Schedule the drift check** (cron, GitHub Actions or Airflow), weekly.
2. **Alerting:** notify someone (email or Slack) when PSI crosses a threshold.
3. **Fix the drift reference:** use the natural production mix, not the balanced
   training mix.
4. **Request / prediction store:** the API keeps nothing after it answers. Append
   each request and prediction to a per-week parquet under `data/live/` (or
   SQLite/Postgres) so drift can run on real traffic.
5. **Push-based ingestion:** a worker that classifies and risk-scores every
   complaint on arrival, instead of waiting for `/classify` calls.
6. **Automated retraining** on drift or on a schedule.
7. **Champion/challenger gate:** never auto-promote a retrain. Score it on the
   frozen test set, promote only if it beats the incumbent on macro-F1 / PR-AUC,
   and keep the old model for rollback.
8. **Outcome feedback:** record whether the company actually paid, to measure
   live accuracy and not just input drift.
9. **Calibrate probabilities** (Platt or isotonic) for `/risk` and `/classify`.
10. **API hardening:** auth, rate limiting, request-size caps, and a batch
    `/classify` endpoint.
11. **Model versioning:** register served models in the MLflow registry and tie
    each one to its training run.
