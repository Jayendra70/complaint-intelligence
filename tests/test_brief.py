"""Retrieval, offline brief and groundedness audit. No API key needed."""

import pytest

from src import config
from src.brief import audit, generate, retrieval

needs_topics = pytest.mark.skipif(
    not config.TOPICS_PARQUET.exists(),
    reason="needs data/processed/complaint_topics.parquet (run notebooks 00, 01, 06)",
)


@pytest.fixture(scope="module")
def topics():
    df = retrieval.load_topics()
    return df, retrieval.build_topic_labels(df), retrieval.find_spikes(df)


@needs_topics
def test_spike_count_matches_topic_report(topics):
    _, _, spikes = topics
    assert len(spikes) == 54


@needs_topics
def test_evidence_has_real_ids(topics):
    df, labels, spikes = topics
    evidence = retrieval.build_evidence(df, "2023-01-16", labels, spikes)
    assert len(evidence["topics"]) > 0
    for topic in evidence["topics"]:
        assert topic["complaints"]
        for c in topic["complaints"]:
            assert isinstance(c["complaint_id"], int)


@needs_topics
def test_offline_brief_is_grounded(topics):
    df, labels, spikes = topics
    evidence = retrieval.build_evidence(df, "2023-01-16", labels, spikes)
    result = generate.generate_brief(evidence, offline=True)
    report = audit.audit_brief(result["text"], retrieval.all_valid_ids(evidence))
    assert report["hallucinated_count"] == 0
    assert report["passed"] is True


def test_audit_catches_a_fake_id():
    report = audit.audit_brief("The sky is falling. [ID: 9999999]\n", {6466819, 6455878})
    assert 9999999 in report["hallucinated_ids"]
    assert report["passed"] is False


def test_audit_flags_uncited_claims():
    text = (
        "## Trend\n"
        "- Mortgage complaints rose sharply this week. [ID: 6466819]\n"
        "- This will keep getting worse next month.\n"
    )
    report = audit.audit_brief(text, {6466819})
    assert report["hallucinated_count"] == 0
    assert report["unsupported_claim_lines"] == 1
    assert report["passed"] is False
