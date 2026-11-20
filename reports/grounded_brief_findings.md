# Grounded weekly brief

A weekly complaint brief for an operations team. Each week the pipeline finds the
topics that spiked, pulls representative real complaints for each, and writes a
short summary (headline / trend / evidence / recommended action) in which every
factual line cites complaint IDs. A separate audit checks that no cited ID was
invented and no claim is left without a citation.

Code: `src/brief/`.

## How it works

1. **Retrieval (`retrieval.py`).** Loads the topic parquet from notebook 06 and
   rebuilds a short keyword label per topic with class-based TF-IDF (the same
   idea as BERTopic's c-TF-IDF). It then re-detects spiking topic-weeks with the
   rule from `topics_report.md` (≥ 2.0 std above the trailing 6-week mean,
   ≥ 15 complaints, ≥ 3 weeks of history), which reproduces the same 54 spikes.
   For each spiking topic it returns the top-k complaints ranked by how many of
   the topic's keywords they contain. Sentence embeddings weren't stored, so
   keyword overlap stands in for similarity.
2. **Generation (`generate.py`).** Formats the evidence and asks a free Groq
   model (the first available in `GROQ_MODELS`) for the structured brief. The
   prompt forbids inventing IDs or facts and requires a citation on every line,
   including the headline and actions. Groq is OpenAI-compatible, so switching
   providers means changing the base URL, key and model list. An offline
   template mode builds a grounded brief with no API call; the tests and the two
   template briefs use it.
3. **Audit (`audit.py`).** Pulls every `[ID: ...]` citation, flags any ID that
   isn't in the evidence, flags factual lines with no citation, and reports the
   unsupported-claim rate. Target: 0.

## Groundedness results

| Brief | Mode | Citations | Hallucinated IDs | Unsupported-claim rate | Result |
|---|---|---|---|---|---|
| week of 2023-01-16 (`sample_weekly_brief.md`) | template | 21 (9 unique) | 0 | 0.0 | PASS |
| week of 2023-05-08 (`weekly_brief_2023-05-08.md`) | template | 21 (9 unique) | 0 | 0.0 | PASS |
| week of 2023-01-16 (`weekly_brief_2023-01-16.md`) | LLM (Groq) | 30 (15 unique) | needs that week's evidence pack; rerun locally | 0.455 (5 of 11 lines) | FAIL |

The LLM brief's 5 uncited lines are its headline and its recommended actions.
That brief was generated before the prompt required a citation on every line;
it hasn't been regenerated since. The audit caught exactly the failure it was
built for.

The audit is also tested on bad input: a brief citing a fake ID (`9999999`) and
an uncited claim are both marked FAIL (`tests/test_brief.py`).

## Cost and latency

- **Offline mode:** $0, ~0 s, no tokens, deterministic.
- **Groq mode:** the free tier needs no card, so cost is $0 (rate-limited, not
  billed). One brief is a single chat call over ~4 topics × 5 complaints
  (~2–3k prompt tokens, a few hundred completion tokens). The runner prints token
  counts and wall-clock latency. For a paid model, set `USD_PER_1K_TOTAL_TOKENS`
  in `run_weekly_brief.py` and the same line reports dollars.

## How to run

```
# offline (no key needed):
python -m src.brief.run_weekly_brief --offline --week 2023-01-16

# LLM prose (free Groq key):
export GROQ_API_KEY=your_key          # from https://console.groq.com
python -m src.brief.run_weekly_brief --week 2023-01-16

# show what would be retrieved, no brief:
python -m src.brief.run_weekly_brief --dry-run --week 2023-01-16

pytest tests/test_brief.py
```

## A real issue it surfaced

In the week of 2023-01-16 the largest spike (z = 21.5) is a cluster of Wells Fargo
loan-modification denials, with a second Wells Fargo cluster (z = 16.7, 146
complaints) about accounts and money. This lines up with the CFPB's December 2022
enforcement action against Wells Fargo: an external event showing up as a
complaint spike.

## Limitations

- **Representative complaints are chosen by keyword overlap, not embedding
  similarity.** The notebook 06 embeddings weren't persisted. Storing them would
  allow a nearest-to-centroid pick.
- **Topic labels are unsupervised** and shift with the seed (see
  `topics_report.md`). The brief names topics by keywords, not curated names.
- **The audit checks that cited IDs exist and that lines carry citations.** It
  doesn't verify that each cited complaint supports its sentence; that still
  needs a manual spot-check.
- **Only ~26 weeks of history**, so the spikes are illustrative.
