# Data Card: CFPB Consumer Complaint Database

## Overview
- **Source:** US Consumer Financial Protection Bureau (CFPB)
- **URL:** [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **License:** US Government Public Domain
- **Full scale:** millions of complaints since 2011; 1M+ include consumer narratives (opt-in since 2015)

## Six-month dataset (behind every reported result)
Built by `notebooks/00_data_download.ipynb` and `notebooks/01_data_cleaning.ipynb`; all numbers below are printed by notebook 01.

| Step | Rows |
|---|---|
| CFPB complaints with a narrative, Jan 1 – Jun 30 2023 | 234,719 |
| After removing near-duplicate template narratives | 173,468 (61,251 removed) |
| Temporal split: train, 2023-01-01 → 2023-05-24 | 138,774 |
| Temporal split: test (newest 20%), 2023-05-24 → 2023-06-30 | 34,694 |
| Training sample balanced by product (cap 8,000 per class, rare classes kept whole); both tasks train on it | 47,121 |

- The natural-mix train set (138,774 rows) is kept as `train_natural.parquet` for the transformer's validation set; the test set is never balanced.
- Monetary-relief label: 3.0% positive in test (1,041 of 34,694), 7.25% in the 47,121-row training sample. Before de-duplication: 5,751 monetary relief, 1,378 untimely, 7,038 escalated.
- Six-month EDA (`reports/eda_report.md`): median narrative about 149 words, 95% under 619 words; 82.5% contain `XXXX` masks.

## Earlier snapshot (March 2023, used during development)
- `cfpb_complaints_2023-03.csv` — real export, **71,236 complaints**, 2023-03-01 -> 2023-03-31 (+5 early-April rows)
- Narratives present: **26,285 (36.9%)**
- After cleaning + near-duplicate removal: **19,678 rows** -> train 15,742 / test 3,936 (temporal split at 2023-03-27)
- Escalation proxy label positive rate: **3.1%** raw narrative subset, **4.1%** train after dedup
- **Not in this export:** `Consumer disputed?` (discontinued 2017-04) and `Consumer consent provided?`
- Longer windows / the full 1M+ narrative dataset: run `notebooks/00_data_download.ipynb` (API or bulk mode)

## Canonical fields (after `normalize_columns`)
- `date_received` — when the CFPB received the complaint (ISO timestamps in API exports)
- `product` / `sub_product`, `issue` / `sub_issue` — complaint taxonomy (labels for Task A)
- `consumer_complaint_narrative` — the free-text narrative (the model input)
- `company`, `state`, `zip_code` — metadata
- `company_response_to_consumer`, `timely_response` — **label-source fields for Task B; dropped from processed data (leakage guard)**
- `complaint_id` — stable ID, used later for grounded-brief citations

## Known wrinkles (measured, not assumed)
1. **Narrative opt-in bias:** only 36.9% of complaints carry a narrative; writers are plausibly angrier/more articulate than average. All models learn this subset.
2. **Discontinued dispute flag:** `Consumer disputed?` ended April 2017 -> proxy label; see `reports/target_decision_memo.md`.
3. **Extreme class imbalance:** top product = 65.5% of the narrative subset (9 classes); escalation positives ~4%.
4. **Redaction tokens:** 85.7% of narratives contain `XXXX` PII masks (median 11 per affected narrative). Kept as-is for now; revisited in error analysis.
5. **Boilerplate duplicates:** ~25% of narratives are near-identical templates (credit-repair letters); removed via normalized-text dedup, earliest occurrence kept.
6. **Unresolved complaints:** rows with response "In progress" have an undefined outcome and are excluded (0 rows in this snapshot; matters on full data).

## Versioning
- v0 (initial): schema + assumed wrinkles
- v0.2: real measured statistics from the 2023-03 snapshot; leakage guard documented
- **v0.3 (current):** six-month dataset funnel added; it is the data behind every reported result
