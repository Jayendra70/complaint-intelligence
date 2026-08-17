# Complaint Intelligence Platform

An end-to-end NLP system over real CFPB consumer complaints: classifies every
complaint, predicts escalation risk, detects emerging issues before humans
notice them, and (later) writes grounded, citation-backed executive briefs.

> **Status: Phase 1 of 7 complete** — data pipeline, cleaning, temporal split
> and full EDA are done on real data (26k narratives, March 2023 window).
> Baselines (Phase 2) are next. Roadmap: see `02_roadmap.md` in the project docs.

## Current results

No model results yet — and that is deliberate. The TF-IDF baseline (Phase 2)
gets locked before any transformer is trained, so every deep learning number
reported later has an honest benchmark to beat.

What exists today:

| Artifact | Where |
|---|---|
| Real dataset (71,236 complaints, 26,285 narratives) | `data/raw/` (gitignored) |
| Cleaning + temporal split pipeline | `notebooks/01_data_cleaning.ipynb` |
| Full EDA with 6 figures | `notebooks/02_eda.ipynb` -> `reports/eda_report.md` |
| Escalation-target design decision | `reports/target_decision_memo.md` |
| Data documentation | `data/DATA_CARD.md` |

## Key dataset facts (from the EDA)

- Narrative opt-in rate **36.9%** -> selection bias, documented
- **25%** of narratives are near-duplicate boilerplate -> deduplicated
- Top product class holds **65.5%** of rows -> macro metrics + class weights mandatory
- Escalation label positive rate **~4%** -> PR-AUC and calibration, not accuracy
- p95 narrative ≈ **564 words (~790 tokens)** -> 512-token truncation is a real decision

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order:

1. `00_data_download.ipynb` — pull real CFPB data (API window or full bulk dump)
2. `01_data_cleaning.ipynb` — clean, label, dedup, temporal split -> parquet
3. `02_eda.ipynb` — regenerate `reports/eda_report.md` + all figures
4. `03`–`06` — Phases 2–4 (baselines, transformers, error analysis, topics)

## Design decisions worth reading

- **Escalation target:** the CFPB dropped `Consumer disputed?` in 2017; the
  proxy-label decision and its tradeoffs live in `reports/target_decision_memo.md`.
- **Leakage guard:** the response/timeliness fields that *define* the label are
  physically dropped from the processed parquet files.
- **Temporal split:** train on the past, test on the future — mirrors production.

## Limitations (honest section)

- Current window is one month (March 2023); the bulk download path in
  notebook 00 scales this to the full 1M+ narrative dataset.
- Proxy escalation label approximates severity, it does not measure consumer
  dissatisfaction directly (see memo).
- Narrative subset only — conclusions do not transfer to complaints without text.
