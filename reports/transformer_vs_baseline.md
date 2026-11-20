# Transformer vs baseline (6-month data)

A fine-tuned DistilBERT + LoRA compared against the TF-IDF + Logistic Regression
baseline on the same data: what went wrong in the first run, what changed after
fixing it, and why the baseline still ships.

## Setup

- **Data:** CFPB complaints, Jan–Jun 2023. The product-task training data was
  class-balanced by sampling (cap ~8,000 per class). The test set (34,694 rows)
  keeps its natural class mix.
- **Comparison:** both models are scored on the same natural test set, with a
  temporal split (train on older complaints, test on newer).
- **Metrics:** macro-F1 for product (equal weight per class) and PR-AUC for the
  ~3%-positive monetary task. Accuracy and ROC-AUC are reported but not used to
  pick models, since both look good under this much imbalance.

## Headline

| Task | Baseline (TF-IDF + LogReg) | Transformer, Tier 1 run | Winner |
|---|---|---|---|
| Product (macro-F1) | 0.7065 | **0.7386** | transformer, +3.2 pts |
| Monetary (PR-AUC) | **0.3337** | 0.3308 | baseline (a tie) |

The first transformer run lost the product task. A corrected second run wins it.

## Run 1: why the first transformer lost

The first DistilBERT + LoRA run (rank 8) scored product macro-F1 0.6832, 2.3
points below the baseline's 0.7065, and monetary PR-AUC 0.3275. The baseline won
8 of 9 product classes.

Two problems in that run were holding the model back:

1. **Imbalance corrected twice.** The product training data was balanced by
   sampling *and* trained with class weights. That pushed the model too far
   toward rare classes: high recall, low precision (payday precision 0.31,
   vehicle 0.42), which drags macro-F1 down.
2. **Balanced validation, natural test.** Early stopping used a balanced
   validation set while the test set is natural, so validation overstated the
   model (val macro-F1 0.79 vs test 0.68) and model selection targeted the wrong
   distribution.

Run 2 tested whether these were the reason it lost.

## Run 2 (Tier 1): what changed

Three changes, made together:

- **One imbalance correction.** Plain cross-entropy for product (the data is
  already balanced); class weights kept only for monetary, whose data is not
  balanced.
- **Natural-mix validation set**, so early stopping matches the test
  distribution.
- **LoRA rank 8 → 16.**

Result: product macro-F1 **0.7386** (val 0.7473, plain-CE config) and monetary
PR-AUC **0.3308** (val 0.3548, focal config). Product moved 5.5 points
(0.6832 → 0.7386), from below the baseline to 3.2 points above it.

Caveats:

- **Single run, no seed-variance check.** Macro-F1 moves around on the rare
  classes, so read this as "wins by about 3 points", not "definitively better".
- **The three changes are bundled**, so the 5.5-point swing can't be attributed
  to any one of them. Dropping the double correction is the most likely main
  driver, but this run doesn't prove it.
- **No per-class breakdown was saved from the Colab run.** Notebook 05
  re-scores the saved adapter class by class (`reports/error_analysis_report.md`).

## Monetary relief: a tie

| model | PR-AUC | ROC-AUC |
|---|---|---|
| Baseline (LogReg word+char) | **0.3337** | 0.9396 |
| Transformer (Tier 1, focal) | 0.3308 | 0.9401 |

The baseline is marginally ahead. Both are about 10× the random floor (~0.03), so
monetary relief is partly predictable from the text. For the business, the
useful view is a triage table (review the top X% riskiest complaints and catch
most of the monetary-relief cases) rather than AUC on its own. The Run 2 changes
did not move this task.

## Decision: ship the baseline

The transformer now wins product by about 3 points, but the production model
stays TF-IDF + Logistic Regression:

- **No gain on the task that matters operationally.** Monetary-relief triage is
  a tie.
- **The product gain is modest and unreplicated.** +3 macro-F1 from one run with
  no seed check.
- **The baseline is much cheaper to run.** One scikit-learn pickle scores a
  median-length complaint in about 2.8 ms on a 2-thread CPU (under 1 ms per
  complaint in batches; `reports/latency_report.md`). The transformer needs
  PyTorch + Transformers + PEFT and a GPU to be fast. For a triage service, that
  extra cost isn't justified by a 3-point product gain and no monetary gain.

The baseline is what `src/serving/` serves. The Tier 1 transformer is kept as a
documented benchmark (`models/*_distilbert_lora/`).

## Why TF-IDF is hard to beat here

Product classification is mostly lexical. Narratives say "mortgage",
"repossession", "student loan servicer" or "credit report" almost verbatim, and
TF-IDF with character n-grams picks that up directly. A transformer's strengths
(word order, composition, long-range context) help less when the label is
practically named in the text. Truncating to 256 tokens also drops the tail of
long narratives, which TF-IDF reads in full. Careful tuning gets the transformer
ahead, but only by a small margin.

## Open work

1. Seed-variance check: rerun Tier 1 with 2–3 seeds to confirm the +3 points.
2. DeBERTa-v3-base comparison: left out of scope, since the transformer isn't
   shipped either way.
