# Complaint Intelligence Platform

An end-to-end NLP system over real CFPB consumer complaints: classifies every
complaint by product, scores monetary-relief risk, detects emerging issues
before humans notice them, writes grounded citation-backed weekly briefs, and
serves all of it behind a FastAPI service with drift monitoring.

> **Status: Phase 6 of 7 complete** — data pipeline, EDA, baseline, DistilBERT+LoRA
> transformer, emerging-issue topics, grounded LLM brief, and now a running
> FastAPI service (`/classify`, `/risk`, `/topics/trends`, `/brief/latest`) with
> drift monitoring, smoke tests, and Docker. **Next: Phase 7** — Streamlit
> dashboard + README polish (architecture diagram, demo GIF, rehearsed
> walkthrough). Roadmap: `02_roadmap.md` in the project docs.

## Current results (6-month data, Jan–Jun 2023, temporal split)

Test set = 34,694 rows at the natural class mix (never balanced). The baseline is
the **shipped** model; the transformer was improved but not shipped on cost/benefit
(see `reports/phase3_findings.md`).

| Task | Baseline (shipped) | DistilBERT+LoRA (Tier 1) |
|---|---|---|
| Product classification (macro-F1) | **0.7065** | 0.7386 |
| Monetary-relief risk (PR-AUC) | **0.3337** | 0.3308 |

Emerging-issue layer: 40 topics, 54 topic-week spikes, seed-stability ARI 0.755
(`reports/topics_report.md`). Weekly briefs are grounded with mandatory
complaint-ID citations and pass a 0-hallucination audit (`reports/phase5_findings.md`).

## Run the API (Phase 6)

The API loads two saved baseline models. They are not committed as source — build
them once (~8 min on a laptop CPU), then start the service:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # or requirements-serving.txt for the API only
python -m src.serving.train_baseline     # writes models/*.joblib + drift_reference.json
uvicorn src.serving.app:app --reload     # http://localhost:8000/docs
```

Or with Docker (one command):

```bash
docker compose up --build                # API at http://localhost:8000/docs
```

Drift check and tests:

```bash
python -m src.monitoring.drift --current data/processed/test.parquet
pytest                                   # 8 smoke tests
```

## Notebooks (the modelling pipeline, in order)

1. `00_data_download.ipynb` — pull real CFPB data
2. `01_data_cleaning.ipynb` — clean, label, dedup, temporal split → parquet
3. `02_eda.ipynb` — regenerate `reports/eda_report.md` + figures
4. `03_baseline.ipynb` — Phase 2 baseline (locked)
5. `04_transformer_colab.ipynb` — Phase 3 DistilBERT+LoRA (Colab T4)
6. `05_error_analysis.ipynb` — per-class F1, disagreements, truncation, memorization probe
7. `06_topics_over_time.ipynb` — Phase 4 emerging-issue detection

## Design decisions worth reading

- **Escalation target:** the CFPB dropped `Consumer disputed?` in 2017; the
  proxy-label decision and its tradeoffs live in `reports/target_decision_memo.md`.
- **Leakage guard:** the response/timeliness fields that *define* the label are
  physically dropped from the processed parquet files.
- **Temporal split:** train on the past, test on the future — mirrors production.
- **Ship the cheaper model:** the transformer edged the baseline on product but
  tied on monetary; the baseline is a sub-ms CPU model, so it ships. Improving a
  model and still declining to deploy it is the judgment on display.

## Limitations (honest section)

- Only ~26 weeks of history, so emerging-issue spikes are illustrative, not
  battle-tested.
- The `/classify` confidence is not calibrated (balanced LogReg is over-confident);
  use it for routing, not as a true probability.
- Drift monitoring's product reference is the (balanced) training mix, so natural
  production data legitimately shows high product PSI — set the reference to the
  production mix for real use (see `reports/phase6_findings.md`).
- Narrative subset only — conclusions do not transfer to complaints without text.

## Future work (production hardening)

The service does detection honestly and leaves the automation as documented next
steps: scheduled + alerting drift checks, logging live requests to a store so PSI
runs on real traffic (the API currently persists nothing after it answers), a
push-based worker that auto-classifies every incoming complaint, an automated
retraining pipeline with a champion/challenger gate (never promote a retrain
unless it beats the incumbent on a frozen test set), calibrated risk
probabilities, API auth/rate-limiting, and MLflow model versioning. The complete, phase-by-phase
backlog (with a priority shortlist) lives in `FUTURE_WORK.md`; the Phase 6 detail
is in `reports/phase6_findings.md`.
