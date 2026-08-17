# Phase 3 Findings — Transformer vs Baseline (6-month data)

*What we set out to test, what went wrong the first time, what we found after
fixing it, and what we decided. This is the Phase 3 conclusion: a fine-tuned
DistilBERT + LoRA benchmarked honestly against a TF-IDF + Logistic Regression
baseline on the same data.*

## Setup

- **Data:** CFPB 6-month file (Jan–Jun 2023). Product-task training data was
  class-balanced by sampling (cap ~8,000/class); the **test set (34,694 rows)
  was left at its natural class mix** so every score below is honest.
- **Fair comparison:** both models are scored on the **same natural test set**.
  Temporal split (train older, test newer).
- **Metrics:** macro-F1 for product (equal weight per class), PR-AUC for the
  ~3%-positive monetary task. Accuracy / ROC-AUC are reported but not trusted
  under this imbalance.

## The headline

| Task | Baseline (TF-IDF + LogReg) | Transformer, Tier 1 run | Winner |
|---|---|---|---|
| Product (macro-F1) | 0.7065 | **0.7386** | transformer, +3.2 pts |
| Monetary (PR-AUC) | **0.3337** | 0.3308 | baseline (a tie) |

This is a **change from our first conclusion.** The first transformer run *lost*
the product task; a fixed second run *wins* it. The story below is that arc,
because the arc — not the single number — is the interview-worthy part.

## Run 1 — the naive transformer, and why it lost

The first DistilBERT + LoRA run (rank 8) scored **product macro-F1 0.6832** —
2.3 points *below* the baseline's 0.7065 — and **monetary PR-AUC 0.3275**, a hair
below the baseline. Per class, the baseline won 8 of 9 product classes. It looked
like a clean "deep learning wasn't worth it" result.

But two mechanical bugs in that run were suppressing the model:

1. **Double imbalance correction.** The product training data was class-balanced
   *by sampling* **and** trained with class weights. Correcting twice over-pushed
   the model toward rare classes — high recall, low precision (payday precision
   0.31, vehicle 0.42) — which drags macro-F1 down.
2. **Balanced validation, natural test.** Early-stopping was judged on a
   *balanced* validation set while the test set is natural, so validation
   flattered the model (val macro-F1 0.79 vs test 0.68) and model selection was
   optimized for the wrong distribution.

Both were written up as known weaknesses. The honest question was: are they the
reason it lost, or is the baseline just better here? That is what Run 2 tested.

## Run 2 — the Tier 1 fix, and why it won

The second run changed three things together:

- **One imbalance correction, not two** — plain cross-entropy on the product task
  (the data is already balanced by sampling), class weights kept only on the
  monetary task (whose data is *not* balanced).
- **A natural-mix validation set** — so early-stopping is judged on the same
  distribution as the test set.
- **LoRA rank raised 8 → 16** — a little more adapter capacity.

Result: **product macro-F1 0.7386** (val 0.7473, plain-CE config) and **monetary
PR-AUC 0.3308** (val 0.3548, focal config). The product score jumped 5.5 points
(0.6832 → 0.7386) and crossed from *below* the baseline to *3.2 points above* it.

**Honest caveats — state these, don't hide them:**

- It is a **single run with no seed-variance check.** Macro-F1 wobbles on the
  rare classes, so treat this as "wins by ~3 points," not "definitively better."
- The **three changes are bundled**, so the +5.5-point swing can't be cleanly
  attributed to any one of them. The most likely main driver is dropping the
  double correction, but this run doesn't prove that.
- A **per-class Tier 1 breakdown was not saved** from the Colab run. That is
  exactly what the error-analysis notebook (05) produces — it re-scores the saved
  adapter class by class.

## Result — monetary relief: still a tie

| model | PR-AUC | ROC-AUC |
|---|---|---|
| Baseline (LogReg word+char) | **0.3337** | 0.9396 |
| Transformer (Tier 1, focal) | 0.3308 | 0.9401 |

Effectively tied; the baseline is a nose ahead. Both are ~10× the random floor
(~0.03), so the signal is real — monetary relief *is* partly predictable from
text. The business-useful framing is the triage table (review the top X% riskiest
complaints and catch most of the true monetary-relief cases); AUC alone is not the
headline. The fix did **not** move this task.

## Decision — still ship the baseline (and here is the honest reason)

Even though the fixed transformer now **wins the product task by ~3 points**, the
production model stays the **TF-IDF + Logistic Regression baseline.** The reasoning:

- **No gain where it would matter operationally.** The monetary-relief task — the
  one framed as a business triage tool — is a tie. The transformer buys nothing
  there.
- **The product gain is small and unreplicated.** +3 macro-F1 points from a single
  run with no seed check is a real but modest edge, not a decisive one.
- **The baseline is far cheaper to operate.** It is a small scikit-learn model that
  scores a complaint in well under a millisecond on CPU, serialized as one pickle.
  The transformer needs the full PyTorch + Transformers + PEFT stack and a GPU to
  be fast; on CPU it is much slower. For a triage/routing service, that is a large
  jump in serving cost and complexity for a 3-point product gain and no monetary
  gain.

So the baseline ships and is the Phase 6 serving target. The Tier 1 transformer
is retained as a documented, benchmarked comparison.

**Why this is the senior result:** the story is no longer "deep learning lost."
It is *"naive fine-tuning lost to a strong classical baseline; disciplined
imbalance handling made the transformer beat it on product; and it still isn't
worth shipping, because the gain doesn't cover the serving cost and there's no
gain on the second task."* Being able to improve the model **and** decline to ship
it on a cost/benefit basis is the judgment employers are actually screening for —
more than a model that squeaks ahead on one metric.

## Why TF-IDF is so hard to beat here (the underlying lesson)

Product classification is a **lexical, keyword-driven** task. Narratives say
"mortgage," "repossession," "student loan servicer," "credit report" almost
verbatim. TF-IDF with character n-grams captures that signal directly and in full;
a transformer's advantages — word order, composition, long-range context — barely
help when the label is essentially named in the text, and 256-token head
truncation throws away the tail of long narratives that TF-IDF reads whole. The
transformer *can* edge ahead with careful tuning, but it has to work for a small
gain the baseline gives away for free.

## Future / open work

1. **Row-level error-analysis notebook (05)** — per-class Tier 1 breakdown,
   disagreement examples where each model is right, a narrative-length-vs-error
   slice (does 256-token truncation hurt?), and a company-name memorization
   ablation. *This is the next Phase 3 deliverable.*
2. **Seed-variance check on the product win** — rerun Tier 1 with 2–3 seeds to
   confirm the +3 points is stable, if a cleaner claim is wanted for the resume.
3. **DeBERTa-v3-base** comparison — out of scope by decision.
