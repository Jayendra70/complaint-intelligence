# Phase 5 - Grounded LLM Weekly Brief

*Written 2026-08-14. Covers the retrieval -> generation -> groundedness-audit
pipeline in `src/brief/`.*

## What this phase delivers

A weekly "complaint brief" for an operations team. Each week the system finds
which complaint topics spiked, pulls the most representative real complaints for
each, and writes a short exec summary (headline / trend / evidence / recommended
action) in which **every factual line cites real complaint IDs**. A separate
audit then checks that no cited ID was invented and no claim is left unsupported.

The point of the phase is discipline, not vibes: the LLM is only allowed to talk
about complaints it was handed, and we measure whether it obeyed.

## How it works

1. **Retrieval (`retrieval.py`)** - loads the Phase 4 topic parquet, rebuilds a
   short keyword label per topic with a class-based TF-IDF (same idea as
   BERTopic's c-TF-IDF), and re-detects spiking topic-weeks using the exact
   Phase 4 rule (>= 2.0 std above the trailing 6-week average, >= 15 complaints,
   >= 3 weeks of history). This reproduces the locked Phase 4 result of **54
   spikes** precisely. For each spiking topic it returns the top-k complaints,
   ranked by how many of the topic's keywords they contain (a simple, honest
   stand-in for similarity, since sentence embeddings were not stored).

2. **Generation (`generate.py`)** - formats the evidence and asks a free Groq
   model (`llama-3.3-70b-versatile`) for the structured brief, with a system
   prompt that forbids inventing IDs or facts. There is a single clearly-marked
   swap point, so moving to another free provider (e.g. Gemini) is a one-line
   change. An **offline template mode** builds a grounded brief with no API call
   at all - used for the audit, the tests, and the two sample briefs below.

3. **Audit (`audit.py`)** - pulls every `[ID: ...]` citation, flags any ID not
   in the evidence as a hallucination, flags factual lines with no citation, and
   reports the unsupported-claim rate. Target = 0.

## Groundedness results

The two offline (template) briefs pass. The one LLM-written brief fails on
uncited lines:

| Brief | Mode | Citations | Hallucinated IDs | Unsupported-claim rate | Result |
|---|---|---|---|---|---|
| week of 2023-01-16 (`sample_weekly_brief.md`) | template | 21 (9 unique) | 0 | 0.0 | PASS |
| week of 2023-05-08 (`weekly_brief_2023-05-08.md`) | template | 21 (9 unique) | 0 | 0.0 | PASS |
| week of 2023-01-16 (`weekly_brief_2023-01-16.md`) | LLM (Groq) | 30 (15 unique) | needs that week's evidence pack; rerun locally | 0.455 (5 of 11 lines) | FAIL |

The LLM brief's 5 uncited lines are its headline and its recommended actions.
So far the clean audits come from the template; on LLM output the audit is doing
its job by catching lines with no citation. Next: require a citation on every
line in the prompt, including the headline and actions, and rerun.

The audit is also negatively tested: a brief citing a fake ID (`9999999`) and a
citation-free claim is correctly marked FAIL with the fake ID listed and the
uncited line flagged (see `tests/test_brief.py::test_audit_catches_a_fake_id`).

## Cost / latency

- **Offline mode:** $0, ~0 s, 0 tokens - deterministic, no network.
- **Groq mode:** Groq's free tier needs no credit card, so the real cost is $0;
  it is rate-limited, not billed. A single brief is one chat call over ~4 topics
  x 5 complaints (~2-3k prompt tokens, a few hundred completion tokens). The
  runner prints exact token counts and wall-clock latency after each run. If you
  ever move to a paid model, set `USD_PER_1K_TOTAL_TOKENS` in `run_weekly_brief.py`
  and the same line reports the dollar cost.

## How to run

```
# offline (no key, always works, used for the samples and tests):
python -m src.brief.run_weekly_brief --offline --week 2023-01-16

# real LLM prose (free Groq key, no card):
export GROQ_API_KEY=your_key          # from https://console.groq.com
python -m src.brief.run_weekly_brief --week 2023-01-16

# just see what would be retrieved, no brief:
python -m src.brief.run_weekly_brief --dry-run --week 2023-01-16

# groundedness + retrieval tests (no key needed):
pytest tests/test_brief.py
```

## A real emerging issue it surfaced

The week of 2023-01-16, the biggest spike (z=21.5) is a cluster of Wells Fargo
**loan-modification denial** complaints, and a second Wells Fargo cluster (z=16.7,
146 complaints) about accounts and money. That lines up with the real December
2022 CFPB action against Wells Fargo - a genuine external event showing up as a
complaint spike, exactly the kind of thing this layer is meant to catch.

## Honest limitations

- **Representativeness is keyword-overlap, not embedding similarity.** Sentence
  embeddings were not persisted in Phase 4, so top-k selection uses how many
  topic keywords a narrative contains. It is transparent but coarser than a true
  nearest-to-centroid pick. Storing the Phase 4 embeddings would improve this.
- **Topic labels are unsupervised** and shift with the seed (Phase 4 caveat).
  The brief names topics by keywords, not curated names.
- **The audit checks that cited IDs exist and that lines carry citations** - it
  does not verify that each cited complaint truly supports the sentence's claim.
  That still needs a human spot-check (the roadmap's "manually audit 20 claims").
- **Only ~26 weeks of history**, so spikes are illustrative, not battle-tested.
