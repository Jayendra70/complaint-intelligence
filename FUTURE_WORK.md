# Future Work — Complaint Intelligence Platform

*Consolidated, phase-by-phase list of everything not built (or built partway) so
far. Last updated 2026-08-17, after Phase 6. Pulled from each phase's own report:
`reports/eda_report.md`, `baseline_report.md`, `phase3_findings.md`,
`model_card.md`, `error_analysis_report.md`, `topics_report.md`,
`phase5_findings.md`, `phase6_findings.md`, and the stretch goals from the original
roadmap (kept outside this repo). Updated 2026-10-11: company-name probe recorded,
latency measured.*

## Status legend

- **OPEN** — not started; a real next step.
- **PARTIAL** — started/built but not finished, run, or wired in.
- **PLANNED** — already scheduled as the next phase (Phase 7).
- **DECLINED** — deliberately out of scope (kept here so the decision is on record).

## Priority shortlist (do these first)

| # | Item | Phase | Why it matters | Status |
|---|---|---|---|---|
| 1 | Persist Phase 4 sentence embeddings | 4 | Unblocks better brief retrieval AND embedding-drift monitoring — one fix, two payoffs | OPEN |
| 2 | Fix the drift reference to the natural production mix | 6 | Current reference is the balanced train set, so it cries "major drift" falsely | OPEN |
| 3 | Log live requests to a store | 6 | Nothing is saved today, so PSI can't run on real traffic | OPEN |
| 4 | Champion/challenger gate before promoting any retrain | 6 | Stops a retrain from silently making things worse | OPEN |
| 5 | Streamlit dashboard + README polish (demo GIF) | 7 | The layer recruiters actually look at | PLANNED |
| 6 | Seed-variance check on the +3-point product win | 3 | Turns "wins by ~3 pts" into a claim you can defend | OPEN |

---

## Phase 1 — Data & EDA

| Item | Detail | Status |
|---|---|---|
| Scale to the full CFPB dataset | Currently the 6-month file (~234k narratives). The bulk-download path in notebook 00 scales this to the full 1M+ dataset; more data steadies rare-class scores and weekly topic counts. | OPEN |
| Model `is_untimely` on 6-month data | Dropped earlier because it was too rare at March scale (`baseline_report.md` says "revisit on the 6-month data"). Worth re-checking now there is more data. | OPEN |
| `XXXX` PII-mask ablation | 82.5% of narratives contain `XXXX`. Test raw-vs-normalized masking to see if it changes model scores. | OPEN |
| Narrative opt-in bias (~37%) | Only motivated/frustrated customers write narratives — a selection bias. Not fixable, but keep stated in any conclusion. | DOCUMENTED |

## Phase 2 — Baseline

| Item | Detail | Status |
|---|---|---|
| Publish the risk triage table | The "review the top X% riskiest → catch Y% of monetary-relief cases" table is the business-facing operating point; surface it in the dashboard/README. | OPEN |
| Calibration curve for the baseline risk model | Show how trustworthy the probability is at each threshold. | OPEN |

## Phase 3 — Transformer (DistilBERT + LoRA)

| Item | Detail | Status |
|---|---|---|
| Per-class Tier 1 breakdown | Produced by notebook 05 — see `error_analysis_report.md` (transformer wins 7/9 classes, biggest gain on Student loan). | DONE |
| Seed-variance check on the product win | Rerun Tier 1 with 2–3 seeds to confirm the +3 macro-F1 points is stable, not noise. | OPEN |
| Company-name memorization probe | Notebook 05: accuracy is no higher on the 10,878 test rows that name the company unmasked (baseline 0.822 vs 0.839, transformer 0.841 vs 0.859) — no name shortcut. Recorded in `model_card.md`. | DONE |
| Calibrate monetary probabilities | Transformer probabilities are over-confident (curve below diagonal); apply Platt/isotonic if the raw number is ever read as a true probability. | OPEN |
| Longer-context handling | 256-token head truncation drops the tail of long narratives (p95 > 512 tokens). Error analysis shows it isn't clearly hurting, but a sliding-window / longer max_length test would close the question. | OPEN |
| DeBERTa-v3-base comparison | Would likely edge DistilBERT, but adds cost for a model that isn't shipped anyway. | DECLINED |

## Phase 4 — Emerging-issue detection (topics)

| Item | Detail | Status |
|---|---|---|
| **Persist the sentence embeddings** | They were not saved. Storing them unblocks embedding-similarity retrieval in Phase 5 AND embedding-centroid drift in Phase 6. Highest-leverage single fix. | OPEN |
| Embed more than the 40k sample | Only 40k of ~173k narratives were embedded for speed; embedding the full set steadies weekly topic counts. | OPEN |
| Curated topic names | Labels are unsupervised keyword strings that shift with the seed; hand-curate names for the top topics for a cleaner dashboard/brief. | OPEN |
| More weeks of history | Only ~26 weeks, so spikes are illustrative; the full dataset extends the history. | OPEN (needs Phase 1 scale-up) |

## Phase 5 — Grounded LLM weekly brief

| Item | Detail | Status |
|---|---|---|
| Embedding-similarity retrieval | Representative complaints are currently chosen by keyword overlap (embeddings weren't stored). Switch to nearest-to-centroid once Phase 4 embeddings are persisted. | OPEN (depends on Phase 4 item 1) |
| Semantic groundedness check | The audit confirms cited IDs exist and lines carry citations — it does NOT verify each cited complaint actually supports the sentence. Add the roadmap's manual 20-claim spot-check, then automate a semantic check. | OPEN |
| Curated topic names in briefs | Same as the Phase 4 naming item — briefs currently name topics by keywords. | OPEN |

## Phase 6 — Serving & Monitoring

*(Full detail in `reports/phase6_findings.md`. The service does honest detection
and deliberately stops before the automation.)*

| Item | Detail | Status |
|---|---|---|
| Schedule the drift check | Runs by hand today; move to cron / GitHub Action / Airflow, weekly, unattended. | OPEN |
| Alerting | Notify a human (email/Slack) off `drift_report.json` when PSI crosses a threshold. | OPEN |
| Fix the drift reference | Reference is the balanced training mix → natural traffic shows false "major" product drift. Set it to the production mix. | OPEN |
| **Request / prediction store** | The API persists nothing after answering, so there is no place a week of live complaints accumulates for PSI. Append each request+prediction to a per-week parquet under `data/live/` (or SQLite/Postgres at scale). *Answers: "where are the weekly complaints stored?" — nowhere yet; this builds it.* | OPEN |
| **Auto-classify ingestion (push)** | Today nothing is classified unless something calls `/classify`. Add a worker/queue consumer that classifies + risk-scores every complaint as it arrives. *Answers: "what if a complaint isn't sent for classification?" — nothing happens; this makes it automatic.* | OPEN |
| Automated retraining pipeline | On drift or on a schedule: pull a fresh labelled slice, re-fit `train_baseline.py`, write new artifacts. | OPEN |
| **Champion/challenger gate** | Never auto-promote a retrain. Score it against the frozen held-out test set; promote only if it beats the incumbent on macro-F1 / PR-AUC. Keep the old model for rollback. | OPEN |
| Concept-drift feedback loop | Capture the eventual real outcome (did the company pay?) to measure live accuracy, not just input drift. | OPEN |
| Embedding-centroid drift | The roadmap listed this; it wasn't built because Phase 4 embeddings weren't stored. Unblocked by the Phase 4 persistence item. | OPEN (depends on Phase 4 item 1) |
| Calibrated risk probabilities | Apply Platt/isotonic so `/risk` returns a true probability; expose `/classify` confidence honestly. | OPEN |
| API hardening | Auth, rate limiting, request-size caps, and a batch `/classify` endpoint for throughput. | OPEN |
| Model versioning | Register served models in the MLflow registry and tie each served model to its run. | OPEN |

## Phase 7 & beyond (roadmap stretch goals)

| Item | Detail | Status |
|---|---|---|
| Streamlit dashboard | 4–5 views: trends, emerging-issue alerts, risk queue, brief viewer, model-metrics. Compose file already has the service stubbed. | PLANNED (Phase 7) |
| README polish | Architecture diagram, demo GIF, quickstart, honest limitations, rehearsed 60-sec walkthrough. | PLANNED (Phase 7) |
| Deploy to HF Spaces / Render | Free-tier public deployment closes the "can you ship it?" gap — strongly recommended. | OPEN (stretch) |
| Multi-task learning | Single encoder, two heads (classification + risk) — a good interview talking point. | OPEN (stretch) |
| Active-learning demo | Model flags low-confidence complaints for a human-review queue. | OPEN (stretch) |
| Power BI dashboard | Alternative to Streamlit to speak the corporate-analytics dialect. | OPEN (stretch) |

---

*How to use this file: the priority shortlist is the honest "if you only do six
things" list. Everything below it is the full backlog, grouped so you can speak
to any phase's gaps in an interview — which, for a portfolio project, is worth as
much as closing them.*
