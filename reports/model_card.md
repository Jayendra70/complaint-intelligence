# Model Card — Complaint Intelligence Classifiers

*Phase 3 deliverable. Reflects the 6-month run, including the Tier 1 transformer
rerun. The shipped model is the TF-IDF + Logistic Regression baseline; the
DistilBERT + LoRA transformer is documented here as an honest, benchmarked
comparison that — after a fix — beats the baseline on product but is still not
worth shipping. Full reasoning: `reports/phase3_findings.md`.*

## Overview

Two text classifiers over CFPB consumer-complaint narratives:

- **Product classifier** — assigns a complaint to one of 9 product categories.
- **Monetary-relief risk model** — estimates the probability that a complaint
  ends in monetary relief to the consumer (binary, ~3% positive).

Both are trained as separate models and evaluated on the same held-out test set.

## Shipped model (decision)

**The production model is the TF-IDF + Logistic Regression baseline** — word+char
n-grams for the product task, word n-grams for the monetary task, both with
`class_weight="balanced"`. The Tier 1 transformer edges it on product (+3 macro-F1
points) but is tied on monetary and much heavier to serve, so the baseline remains
the production model and the Phase 6 serving target. See Limitations and
`reports/phase3_findings.md` for the full cost/benefit reasoning.

## Metrics (6-month, same natural test set)

**Task A — product classification (9 classes)**

| model | macro-F1 | accuracy |
|---|---|---|
| **Baseline: TF-IDF + LogReg (word+char) — SHIPPED** | 0.7065 | 0.8337 |
| Transformer Run 1 (naive: r=8, double correction, balanced val) | 0.6832 | 0.8178 |
| **Transformer Tier 1 (r=16, single correction, natural val)** | **0.7386** | 0.8537 |

The naive run lost to the baseline; the Tier 1 fix wins by ~3 points. Caveat: the
Tier 1 result is a single unreplicated run and the per-class breakdown is pending
the error-analysis notebook (05).

**Task B — monetary-relief risk (binary, ~3% positive; ~1,041 positives in test)**

| model | PR-AUC | ROC-AUC |
|---|---|---|
| **Baseline: TF-IDF + LogReg (word+char) — SHIPPED** | **0.3337** | 0.9396 |
| Transformer Tier 1 (focal loss) | 0.3308 | 0.9401 |

Effectively a tie, baseline a nose ahead. Both are ~10× the random-guess floor
(~0.03), so the risk signal is genuinely real. Operating point: reviewing the
top ~10% riskiest complaints catches the large majority of monetary-relief cases
(exact figures come from the notebook's top-X% table).

Primary metrics are **macro-F1** (product) and **PR-AUC** (monetary). Accuracy and
ROC-AUC are reported but not optimized — under a 65% majority class and a 3%
positive rate they flatter the model.

## Architecture (transformer comparison model — Tier 1 config)

- **Base model:** `distilbert-base-uncased` (66M parameters).
- **Adaptation:** LoRA (`r=16`, `alpha=32`, dropout 0.05) on the attention
  query/value projections (`q_lin`, `v_lin`); only the LoRA weights and the
  classification head are trained (~1% of parameters).
- **Input:** the free-text narrative only. Metadata (company, state, issue) is
  intentionally excluded so the model reasons from language.
- **Tokenization:** WordPiece, `max_length = 256`, head truncation.
- **Loss:** product task uses plain cross-entropy (data already balanced by
  sampling — one imbalance correction); monetary task keeps class-weighted /
  focal loss (its data is *not* balanced). Selected on validation (product on
  macro-F1, monetary on PR-AUC).
- **Training/validation:** balanced-by-sampling training slice; **natural-mix**
  validation set so early-stopping matches the test distribution.

## Training data

- **Source:** CFPB Consumer Complaint Database, 6-month file (Jan–Jun 2023).
- **Product task train:** class-balanced by sampling (cap ~8,000/class), rare
  classes kept in full. **Test: 34,694 rows at the natural class mix** (never
  balanced), so reported scores are honest.
- **Split:** temporal (train on older complaints, test on newer) — mirrors
  production, avoids look-ahead leakage.
- **Leakage guard:** outcome-defining columns (company response, timely
  response) dropped from the processed data; used only to build labels.

## Intended use

- Triage and routing of incoming complaints by product category.
- Prioritizing a human-review queue for likely monetary-relief cases
  ("flag the top X% riskiest").

**Not** intended for automated decisions affecting consumers, eligibility, or any
regulatory determination. A human stays in the loop.

## Limitations

- **Why the baseline still ships despite losing on product:** the transformer's
  win is ~3 macro-F1 points on a single unreplicated run; the monetary task is a
  tie; and the transformer needs the PyTorch + Transformers + PEFT stack and a GPU
  to serve at speed, versus a sub-millisecond CPU scikit-learn model. For a triage
  service that cost is not justified by the gain. This is a deliberate cost/benefit
  decision, not a claim that the baseline is more accurate.
- **Single-run product result:** no seed-variance check yet; treat +3 points as
  a real-but-modest edge, not a decisive one.
- **Miscalibration (monetary transformer):** predicted probabilities are
  over-confident (calibration curve below the diagonal). Fine for *ranking* /
  triage, but the raw probability should not be read as a true probability without
  calibration (Platt/isotonic).
- **Head truncation to 256 tokens** discards the tail of long narratives (p95 >
  512 tokens). Whether that costs accuracy is untested (needs the row-level error
  analysis).
- **Narratives are opt-in (~37%) and PII-masked (`XXXX`)**; a company-name
  memorization check is not yet done (planned in the error analysis).
- **Rare product classes remain the hardest cases** for both models.

## Deferred / future work

- **Row-level error-analysis notebook (05):** per-class Tier 1 breakdown,
  disagreement examples, length-vs-truncation slice, company-name memorization
  ablation.
- **Seed-variance check** on the Tier 1 product win, if a cleaner resume claim is
  wanted.
- **XXXX raw-vs-normalized ablation** and the optional **DeBERTa-v3-base**
  comparison (out of scope by decision).
