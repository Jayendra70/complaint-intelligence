"""
Smoke tests for the Phase 5 brief pipeline.

These run with no API key and no GPU - they check the retrieval, offline brief,
and groundedness audit all behave. Run with:  pytest tests/test_brief.py

The audit tests use hand-written briefs, so they run on a fresh clone. The
retrieval tests need the locally built topics parquet and are skipped without it.
"""

import os

import pytest

from src.brief import retrieval, generate, audit

# data/processed/complaint_topics.parquet is built by notebooks 00 -> 01 -> 06
# and is gitignored, so it only exists on a machine that ran the pipeline.
needs_topics = pytest.mark.skipif(
    not os.path.exists(retrieval.TOPICS_PARQUET),
    reason="needs data/processed/complaint_topics.parquet (run notebooks 00, 01, 06)",
)


def _load():
    df = retrieval.load_topics()
    labels = retrieval.build_topic_labels(df)
    spikes = retrieval.find_spikes(df)
    return df, labels, spikes


@needs_topics
def test_spike_count_matches_phase4():
    """The spike rule should reproduce the 54 spikes from the Phase 4 report."""
    df, labels, spikes = _load()
    assert len(spikes) == 54


@needs_topics
def test_evidence_has_real_ids():
    """Every complaint in the evidence pack has an integer complaint_id."""
    df, labels, spikes = _load()
    evidence = retrieval.build_evidence(df, "2023-01-16", labels, spikes)
    assert len(evidence["topics"]) > 0
    for topic in evidence["topics"]:
        assert topic["complaints"]
        for c in topic["complaints"]:
            assert isinstance(c["complaint_id"], int)


@needs_topics
def test_offline_brief_is_grounded():
    """The offline brief must pass the audit: no hallucinated IDs, no uncited claims."""
    df, labels, spikes = _load()
    evidence = retrieval.build_evidence(df, "2023-01-16", labels, spikes)
    result = generate.generate_brief(evidence, offline=True)
    valid = retrieval.all_valid_ids(evidence)
    report = audit.audit_brief(result["text"], valid)
    assert report["hallucinated_count"] == 0
    assert report["passed"] is True


def test_audit_catches_a_fake_id():
    """A brief citing an ID not in the evidence must be flagged as hallucinated."""
    valid = {6466819, 6455878}
    bad = "The sky is falling. [ID: 9999999]\n"
    report = audit.audit_brief(bad, valid)
    assert 9999999 in report["hallucinated_ids"]
    assert report["passed"] is False


def test_audit_flags_uncited_claims():
    """A factual line with no citation counts as an unsupported claim."""
    text = (
        "## Trend\n"
        "- Mortgage complaints rose sharply this week. [ID: 6466819]\n"
        "- This will keep getting worse next month.\n"
    )
    report = audit.audit_brief(text, {6466819})
    assert report["hallucinated_count"] == 0
    assert report["unsupported_claim_lines"] == 1
    assert report["passed"] is False
