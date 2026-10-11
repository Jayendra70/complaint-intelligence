# Project Structure — Complaint Intelligence Platform

All paths are relative to the repo root — the top-level `complaint-intelligence/`
folder (this repo is the folder you clone; there is no wrapper folder anymore).
*Last refreshed 2026-10-11 (Phases 0–6 complete; the API and 8 of the 14 tests now
run on a fresh clone; latency benchmark and CI added).*

## Tree at a glance

```
complaint-intelligence/
├── README.md
├── requirements.txt                (full dev env: modelling + serving)
├── requirements-serving.txt        (Phase 6 — lean API-only deps, no torch; used by the Docker image)
├── Dockerfile                      (Phase 6 — lean FastAPI image)
├── docker-compose.yml              (Phase 6 — `docker compose up` runs the API; dashboard stubbed for Phase 7)
├── .dockerignore                   (Phase 6 — keeps raw data / mlruns / notebooks out of the image)
├── PROJECT_STRUCTURE.md            (this file)
├── conftest.py                     (puts the repo root on the import path for pytest)
├── .gitignore
├── .github/workflows/tests.yml     (CI — runs pytest on every push; data-dependent tests skip)
├── data/
│   ├── DATA_CARD.md
│   ├── raw/                        (2 CSVs: March file + 6-month file; gitignored)
│   └── processed/
│       ├── train.parquet           (balanced; baseline + transformer train slice)
│       ├── train_natural.parquet   (natural mix; transformer's natural val set)
│       ├── test.parquet            (natural mix, 34,694 rows — the honest test set)
│       └── complaint_topics.parquet (Phase 4 output — topic per complaint; feeds Phase 5 + the API)
├── notebooks/                      (00–06 — the modelling pipeline; see below)
├── reports/
│   ├── eda_report.md, baseline_report.md, target_decision_memo.md
│   ├── transformer_report.md, phase3_findings.md, model_card.md
│   ├── topics_report.md, error_analysis_report.md
│   ├── weekly_brief_*.md           (Phase 5 sample briefs — /brief/latest serves the newest)
│   ├── phase5_findings.md
│   ├── phase6_findings.md          (Phase 6 — serving & monitoring write-up)
│   ├── latency_report.md           (CPU latency of the shipped models — bench_latency.py)
│   ├── mlflow_runs.md              (the Colab MLflow runs for the transformer, as a table)
│   ├── drift_report.md / drift_report.json  (Phase 6 — written by the drift monitor)
│   └── figures/
├── models/
│   ├── product_distilbert_lora/    (Tier 1 LoRA adapter, 9-class — NOT shipped)
│   ├── monetary_distilbert_lora/   (Tier 1 LoRA adapter, binary — NOT shipped)
│   ├── baseline_product.joblib     (Phase 6 — SHIPPED product model; built by train_baseline.py)
│   ├── baseline_monetary.joblib    (Phase 6 — SHIPPED monetary model)
│   └── drift_reference.json        (Phase 6 — "what normal looks like" for the drift monitor)
├── src/
│   ├── brief/                      (Phase 5 — retrieval, generate, audit, run_weekly_brief)
│   ├── serving/                    (Phase 6 — train_baseline.py, app.py, bench_latency.py)
│   └── monitoring/                 (Phase 6 — drift.py)
├── tests/
│   ├── test_brief.py               (Phase 5 — 5 tests; 2 run without data)
│   ├── test_api.py                 (Phase 6 — 7 FastAPI smoke tests; 6 run without data)
│   └── test_drift.py               (Phase 6 — 2 drift smoke tests; need data)
├── dashboard/                      (empty — Phase 7 Streamlit app)
├── mlruns_colab/, mlflow.db        (local MLflow tracking — gitignored, kept off GitHub)
└── venv/                           ⚠️ recreate before running (see the venv note in the project docs)
```

## src/ — the service code (Phases 5–6)

| Path | What it is |
|---|---|
| `src/brief/` | Phase 5 grounded-brief module (retrieval → generate → audit → runner). |
| `src/serving/train_baseline.py` | Fits + SAVES the shipped baseline (word+char TF-IDF → balanced LogReg) for both tasks; also writes `models/drift_reference.json`. Run once before starting the API. |
| `src/serving/app.py` | FastAPI app: `/health`, `/classify`, `/risk`, `/topics/trends`, `/brief/latest`. Loads models + topic spikes once at start-up; without the topics parquet it still starts and `/topics/trends` returns 503. |
| `src/serving/bench_latency.py` | Times the shipped models on CPU (single request and batch); writes `reports/latency_report.md`. |
| `src/monitoring/drift.py` | PSI + KS on narrative length and PSI on predicted-product mix vs the training reference; writes `reports/drift_report.{json,md}`. |

## notebooks/ — the modelling pipeline, in execution order

| Path | Phase | Status |
|---|---|---|
| `00_data_download.ipynb` | 0 | done |
| `01_data_cleaning.ipynb` | 1 | done |
| `02_eda.ipynb` | 1 | done |
| `03_baseline.ipynb` | 2 | done — baseline product 0.7065, monetary 0.3337 |
| `04_transformer_colab.ipynb` | 3 | done — Tier 1 product 0.7386, monetary 0.3308 |
| `05_error_analysis.ipynb` | 3 | done — per-class F1, truncation slice, company-name probe |
| `06_topics_over_time.ipynb` | 4 | done — 40 topics, 54 spikes, ARI 0.755 |

## How to run the service (Phase 6)

```
python -m src.serving.train_baseline     # build models/*.joblib + drift_reference.json (~8 min)
uvicorn src.serving.app:app --reload     # http://localhost:8000/docs
python -m src.monitoring.drift --current data/processed/test.parquet
pytest                                   # 14 tests (8 run on a fresh clone; 6 need data/processed/)
python -m src.serving.bench_latency      # CPU latency of the shipped models
```

## Empty-for-now directories (filled later)

| Path | Fills up in |
|---|---|
| `dashboard/` | Phase 7 — Streamlit app (compose file already has the service stubbed). |

## Planning docs (kept out of the repo on purpose)

The planning / study material (`01_project_overview.md`, `02_roadmap.md`,
`03_study_plan.md`, and the phase study guides) lives in a separate `DL Project new`
folder and in the Claude project — deliberately not part of the GitHub repo.
