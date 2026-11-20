# Phase 6 — Serving & Monitoring (findings)

*Written 2026-08-17. Code in `src/serving/`, `src/monitoring/`, `tests/`, plus
`Dockerfile` / `docker-compose.yml`. Authored and run in the cloud sandbox
(scikit-learn 1.9.0 — same as the pinned Mac version). All 8 tests pass; the API
boots under uvicorn and serves `/docs`. Docker files validate (`docker compose
config` OK) but the image was not built in the sandbox — see the note below.*

## What was built

A real web service around the shipped baseline, not a pile of notebooks.

1. **`src/serving/train_baseline.py`** — the missing link. Phase 2 trained the
   baseline inside notebook 03 but never saved it. This script re-fits the exact
   same model (word+char TF-IDF → class-balanced LogisticRegression, same
   settings as nb03) for both tasks and saves them to `models/`. As a check it
   scores the held-out test set and prints the numbers:

   | Task | This script | Phase 2 locked |
   |---|---|---|
   | Product (macro-F1) | 0.706 | 0.7065 |
   | Monetary (PR-AUC) | 0.3339 | 0.3337 |

   The tiny differences are minor-version rounding (pandas 3.0 vs 2.3 in the
   sandbox); the models are a faithful reproduction of the shipped baseline.
   Artifacts: `models/baseline_product.joblib` (13 MB),
   `models/baseline_monetary.joblib` (5 MB), `models/drift_reference.json`.

2. **`src/serving/app.py`** — FastAPI service, models loaded once at start-up:
   - `GET  /health` — liveness + which artifacts loaded (dashboard/Docker poll this).
   - `POST /classify` — narrative → predicted product + confidence + top-3.
   - `POST /risk` — narrative → monetary-relief probability + a flag at a tunable threshold.
   - `GET  /topics/trends` — the Phase 4 spikes (54), biggest z-score first, with labels.
   - `GET  /brief/latest` — the newest Phase 5 weekly brief as markdown.
   Pydantic schemas validate every request (empty text → HTTP 422, not a crash),
   and everything is auto-documented at `/docs`.

3. **`src/monitoring/drift.py`** — compares a new batch of complaints against the
   training-time reference and reports three numbers: narrative-length PSI,
   narrative-length KS test, and predicted-product-mix PSI. Writes
   `reports/drift_report.json` + `reports/drift_report.md` for the Phase 7
   dashboard. PSI bands: <0.10 none, 0.10–0.25 moderate, >0.25 major.

4. **Tests** — `tests/test_api.py` (6 smoke tests via FastAPI `TestClient`:
   health, classify returns a valid product, risk returns a 0..1 probability,
   empty text is rejected, trends returns the locked 54 spikes sorted, brief
   endpoint returns markdown) and `tests/test_drift.py` (2 tests: train-vs-train
   shows no drift; artificially shortened text triggers the drift flag).
   **`pytest` → 8 passed.**

5. **Docker (re-added at Jayendra's request; it had been dropped)** —
   `Dockerfile` (lean image: `requirements-serving.txt`, i.e. the scikit-learn
   stack only, no torch), `.dockerignore` (keeps raw data / mlruns / notebooks
   out of the image), and `docker-compose.yml` (the `api` service now; a
   commented `dashboard` service ready for Phase 7). One command:
   `docker compose up --build` → API on `localhost:8000`, docs at `/docs`.

## What the drift run actually showed (honest)

Running `python -m src.monitoring.drift --current data/processed/test.parquet`
(the newer, held-out complaints) gave:

- narrative-length PSI = **0.012** (none) — complaint length is stable over time.
- narrative-length KS: statistic 0.034, p ≈ 9e-5 → **flagged**. With 34k rows the
  KS test is oversensitive: a trivially small shift is "statistically
  significant". PSI is the number to trust here; KS is a secondary tripwire.
- predicted-product PSI = **0.79** (major). This is expected, not a bug: the
  reference distribution comes from the *class-balanced* training set, while the
  test set is the *natural* mix (65% credit-reporting). So the mix legitimately
  differs. In a real deployment the reference should be the natural production
  mix; the mechanism is what's being demonstrated, and it correctly fires.

That nuance is itself an interview talking point: a drift monitor is only as
honest as its reference distribution.

## How to run (on the Mac, after recreating the venv)

```
pip install -r requirements.txt          # or: pip install -r requirements-serving.txt (API only)
python -m src.serving.train_baseline     # writes models/*.joblib + drift_reference.json (~8 min)
uvicorn src.serving.app:app --reload     # API at http://localhost:8000/docs
python -m src.monitoring.drift --current data/processed/test.parquet
pytest                                   # 8 smoke tests (brief + serving + drift)
docker compose up --build                # same API, containerised
```

## Known limitations

- The `/classify` confidence can read 1.00 on clear-cut complaints — that is the
  balanced LogReg being over-confident, not calibrated probability. Fine for
  routing; do not quote it as a true probability.
- The Docker image was **not built in the sandbox** (Docker Hub is blocked here,
  so the `python:3.11-slim` base can't be pulled). The Dockerfile is standard and
  `docker compose config` validates; it will build on the Mac. The app is
  independently proven to boot and serve under uvicorn with the identical
  start-up path the container uses.
- Models are pickled with scikit-learn 1.9.0; the serving image pins the same
  version so the pickle loads. Regenerate the models if you bump scikit-learn.

## Future work (production hardening — deliberately NOT built)

Phase 6 does honest detection and stops there. A real production system would add
the automation around it. Listed so it is clear what is missing on purpose:

1. **Schedule the drift check.** Today `drift.py` is run by hand. In production it
   runs on a schedule (cron / GitHub Action / Airflow), weekly, unattended.
2. **Alerting.** When PSI crosses a threshold, notify a human (email/Slack) off
   the `drift_report.json` output instead of relying on someone opening the file.
3. **Fix the drift reference.** The current product reference is the *balanced*
   training mix, so natural traffic shows false "major" drift. Set the reference
   to the natural production distribution.
4. **Request logging / prediction store.** The API currently keeps nothing after
   it answers. To measure drift on *real* traffic (not a static parquet), persist
   every incoming complaint + prediction to a store — simplest: append to a
   per-week parquet under `data/live/`; more robust: SQLite/Postgres; at scale a
   warehouse. The weekly drift job then reads that week's rows.
5. **Auto-classify ingestion (push, not pull).** Today nothing is classified
   unless something calls `/classify`. A small worker/queue consumer would
   classify + risk-score every complaint as it arrives, so routing is automatic.
6. **Automated retraining pipeline.** On drift or on a schedule: pull a fresh
   labelled slice, re-fit `train_baseline.py`, write new artifacts.
7. **Champion/challenger gate (critical).** Never auto-promote a retrained model.
   Score it against the frozen held-out test set and promote only if it beats the
   incumbent on macro-F1 / PR-AUC. Keep the old model for instant rollback.
8. **Concept-drift / feedback loop.** Capture the eventual real outcome (did the
   company actually pay?) to measure live *accuracy*, not just input drift.
9. **Calibrate risk probabilities** (Platt / isotonic) so `/risk` returns a true
   probability, and calibrate/expose `/classify` confidence honestly.
10. **API hardening.** Auth, rate limiting, request-size caps, and a batch
    endpoint (`/classify` over a list) for throughput.
11. **Model versioning.** Register served models in the MLflow registry and tie
    each served model to the run that produced it.

## Roadmap status after Phase 6

Phases 0–6 complete. **Next: Phase 7** — Streamlit dashboard (`dashboard/` still
empty), README architecture diagram + demo GIF, and the rehearsed 60-second
walkthrough. The `docker-compose.yml` already has the dashboard service stubbed
in, ready to uncomment.
