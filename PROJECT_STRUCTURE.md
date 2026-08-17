# Project Structure — Complaint Intelligence Platform

All paths are relative to the repo root `complaint-intelligence/`, which sits
inside the outer folder `DL_Project/`. *Last refreshed 2026-08-14 (Phases 0–4
complete; Tier 1 transformer done; nb05 built).*

## Tree at a glance

```
complaint-intelligence/
├── README.md
├── requirements.txt                (+ Phase 4 block: sentence-transformers, bertopic, umap-learn, hdbscan)
├── PROJECT_STRUCTURE.md            (this file)
├── SESSION_HANDOFF.md              (paste-into-new-chat context; most current status)
├── .gitignore
├── data/
│   ├── DATA_CARD.md
│   ├── raw/                        (2 CSVs: March file + 6-month file; gitignored)
│   └── processed/
│       ├── train.parquet           (balanced; fits the baseline + transformer train slice)
│       ├── train_natural.parquet   (natural mix; source of the transformer's natural val set)
│       ├── test.parquet            (natural mix, 34,694 rows — the honest test set)
│       └── complaint_topics.parquet (Phase 4 output — topic per complaint; feeds Phase 5)
├── notebooks/
│   ├── 00_data_download.ipynb      DONE
│   ├── 01_data_cleaning.ipynb      DONE (produces the parquet splits)
│   ├── 02_eda.ipynb                DONE (regenerates eda_report + figures)
│   ├── 03_baseline.ipynb           DONE — Phase 2, 6-month baseline numbers locked
│   ├── 04_transformer_colab.ipynb  DONE — Phase 3 Tier 1 run (Colab T4)
│   ├── 04_transformer_kaggle.ipynb  Kaggle P100 variant
│   ├── 05_error_analysis.ipynb     BUILT 2026-08-14 — NOT yet run (venv broken)
│   └── 06_topics_over_time.ipynb   DONE — Phase 4 ran, numbers locked
├── reports/
│   ├── eda_report.md
│   ├── baseline_report.md          (Phase 2 numbers)
│   ├── target_decision_memo.md
│   ├── transformer_report.md       (Phase 3 Tier 1 numbers + fair comparison)
│   ├── phase3_findings.md          (Phase 3 narrative + reframed decision)
│   ├── model_card.md               (shipped = baseline; Tier 1 documented)
│   ├── topics_report.md            (Phase 4 — 40 topics, 54 spikes, ARI 0.755)
│   ├── error_analysis_report.md    (written when nb05 is run)
│   ├── sample_weekly_brief.md      (placeholder — filled in Phase 5)
│   └── figures/                    (EDA figs + monetary_calibration.png + topic_spikes.png + token_length_hist.png)
├── models/
│   ├── product_distilbert_lora/    (6-month Tier 1 LoRA adapter, 9-class)
│   └── monetary_distilbert_lora/   (6-month Tier 1 LoRA adapter, binary)
├── src/                            (empty packages — filled in Phases 5–7)
├── dashboard/                      (empty — Phase 7 Streamlit app)
├── tests/                          (empty — Phase 6 smoke tests)
├── mlruns/ + mlruns_colab/ + mlflow.db   (MLflow tracking; gitignored)
└── venv/                           ⚠️ BROKEN — recreate before running notebooks (see SESSION_HANDOFF §6)
```

## Root files

| Path | What it is |
|---|---|
| `README.md` | Front page: pitch, status, dataset facts, quickstart, decisions, limitations. Update the status line as phases complete. |
| `requirements.txt` | Pinned deps. Phase 4 added sentence-transformers, bertopic, umap-learn, hdbscan. |
| `SESSION_HANDOFF.md` | The most current, self-contained status doc — read this first. |
| `.gitignore` | Keeps data, parquet, `mlruns*/`, `mlflow.db`, `venv/`, `_to_delete/`, OS junk out of git. |

## notebooks/ — the pipeline, in execution order

| Path | What it is | Status |
|---|---|---|
| `00_data_download.ipynb` | Gets data (api / bulk / synthetic modes). | done |
| `01_data_cleaning.ipynb` | Raw CSV → clean parquet: filter narratives, proxy label, dedup, temporal split, leakage guard. | done |
| `02_eda.ipynb` | Full EDA; regenerates `eda_report.md` + figures. | done |
| `03_baseline.ipynb` | Phase 2: TF-IDF + LogReg/LinearSVC, metrics, MLflow. Baseline = product 0.7065, monetary 0.3337. | done |
| `04_transformer_colab.ipynb` | Phase 3: DistilBERT + LoRA on Colab T4. Tier 1 run = product 0.7386, monetary 0.3308. | done |
| `05_error_analysis.ipynb` | Phase 3: per-class F1, model disagreements, 256-token truncation slice, company-name memorization probe. Refits the baseline + loads the Tier 1 adapter. | built, not run |
| `06_topics_over_time.ipynb` | Phase 4: MiniLM embeddings → BERTopic → weekly spikes → case studies. | done |

## Empty-for-now directories (filled in later phases)

| Path | Fills up in |
|---|---|
| `src/brief/` | Phase 5 — grounded LLM brief module (retrieval over `complaint_topics.parquet`). |
| `src/serving/`, `src/monitoring/`, other `src/` packages | Phase 6 — modules re-extracted from notebooks for the FastAPI service. |
| `dashboard/` | Phase 7 — Streamlit app. |
| `tests/` | Phase 6 — pytest smoke tests. |

## Outer folder (`DL_Project/`, one level above the repo)

| Path | What it is |
|---|---|
| `01_project_overview.md`, `02_roadmap.md`, `03_study_plan.md` | Planning docs (what/why, 8-week roadmap, study plan). |
| `.DS_Store` | macOS junk; gitignored. |

*(Note: the outer folder was previously named `DL Project new `. It was renamed to
`DL_Project`, which broke the venv's hard-coded paths — see SESSION_HANDOFF §6.)*

## Workflow (short version)

```
00 → 01 → 02 → 03 (baseline) → 04 (transformer, Colab) → 05 (error analysis)
                                         ↘ 06 (topics, Phase 4) → Phase 5 brief
```

1. Recreate the venv (it's broken): `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. Notebooks 00–04 and 06 are already run; outputs are locked.
3. Next: run `05_error_analysis.ipynb` (optional polish), then start Phase 5.
