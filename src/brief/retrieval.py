"""Find spiking topic-weeks and pull representative complaints for the weekly brief.

python -m src.brief.retrieval    # print spike counts and the busiest week's evidence
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src import config

# Spike rule, matching reports/topics_report.md.
TRAILING_WEEKS = 6
MIN_PRIOR_WEEKS = 3
Z_THRESHOLD = 2.0
MIN_COUNT = 15

OUTLIER_TOPIC = -1  # BERTopic's "no topic" bucket

# PII masks and "00" cents fragments would otherwise dominate the topic labels.
PII_STOPWORDS = ["xxxx", "xx", "xxxx xxxx", "xx xx", "xxxx xx", "xx xxxx", "00"]

DEFAULT_TOP_K = 5


def load_topics(path: str | Path = config.TOPICS_PARQUET) -> pd.DataFrame:
    """Read the topics parquet and add a Monday-dated `week` column."""
    df = pd.read_parquet(path)
    naive_date = df["date_received"].dt.tz_localize(None)
    monday = naive_date - pd.to_timedelta(naive_date.dt.weekday, unit="D")
    df["week"] = monday.dt.normalize()
    return df


def build_topic_labels(df: pd.DataFrame, n_words: int = 4) -> dict[int, str]:
    """Label each topic with its top class-based TF-IDF terms (like BERTopic's c-TF-IDF)."""
    real = df[df["topic"] != OUTLIER_TOPIC]
    joined = real.groupby("topic")[config.TEXT_COLUMN].apply(lambda s: " ".join(s.astype(str)))

    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS) + PII_STOPWORDS,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=3,
    )
    matrix = vectorizer.fit_transform(joined.values)
    terms = np.array(vectorizer.get_feature_names_out())

    labels = {}
    for row_index, topic_id in enumerate(joined.index):
        scores = matrix[row_index].toarray().ravel()
        top_positions = scores.argsort()[::-1][:n_words]
        labels[int(topic_id)] = ", ".join(terms[top_positions])
    return labels


def find_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Every topic-week at least Z_THRESHOLD std above its trailing mean, biggest first."""
    real = df[df["topic"] != OUTLIER_TOPIC]
    weekly = real.groupby(["topic", "week"]).size().reset_index(name="count")

    rows = []
    for topic_id, sub in weekly.groupby("topic"):
        sub = sub.sort_values("week").reset_index(drop=True)
        for i in range(len(sub)):
            prior = sub["count"].iloc[max(0, i - TRAILING_WEEKS) : i]
            if len(prior) < MIN_PRIOR_WEEKS:
                continue
            mean = prior.mean()
            std = prior.std()
            # A flat history has no spread to measure a spike against.
            if std == 0 or np.isnan(std):
                continue
            count = sub["count"].iloc[i]
            z = (count - mean) / std
            if z >= Z_THRESHOLD and count >= MIN_COUNT:
                rows.append(
                    {
                        "topic": int(topic_id),
                        "week": sub["week"].iloc[i],
                        "count": int(count),
                        "z_score": round(float(z), 2),
                    }
                )

    spikes = pd.DataFrame(rows, columns=["topic", "week", "count", "z_score"])
    return spikes.sort_values("z_score", ascending=False).reset_index(drop=True)


def spikes_in_week(spikes: pd.DataFrame, week) -> pd.DataFrame:
    week = pd.Timestamp(week).normalize()
    picked = spikes[spikes["week"] == week]
    return picked.sort_values("z_score", ascending=False).reset_index(drop=True)


def representative_complaints(
    df: pd.DataFrame, topic_id: int, week, label: str, top_k: int = DEFAULT_TOP_K
) -> pd.DataFrame:
    """Top-k complaints of a topic-week, ranked by how many label keywords they contain.

    Sentence embeddings weren't stored, so keyword overlap stands in for similarity.
    Ties go to the longer narrative.
    """
    week = pd.Timestamp(week).normalize()
    keywords = [w.strip() for w in label.split(",")]

    cohort = df[(df["topic"] == topic_id) & (df["week"] == week)].copy()
    if cohort.empty:
        return cohort

    lowered = cohort[config.TEXT_COLUMN].astype(str).str.lower()
    score = np.zeros(len(cohort), dtype=int)
    for word in keywords:
        score = score + lowered.str.contains(word, regex=False).astype(int).values
    cohort["match_score"] = score
    cohort["length"] = lowered.str.len().values

    cohort = cohort.sort_values(["match_score", "length"], ascending=False)
    return cohort.head(top_k)


def build_evidence(
    df: pd.DataFrame, week, labels: dict[int, str], spikes: pd.DataFrame, top_k: int = DEFAULT_TOP_K
) -> dict:
    """Evidence pack for one week: {"week": ..., "topics": [{..., "complaints": [...]}]}."""
    week_str = pd.Timestamp(week).normalize().strftime("%Y-%m-%d")

    topics_out = []
    for _, spike in spikes_in_week(spikes, week).iterrows():
        topic_id = int(spike["topic"])
        label = labels.get(topic_id, f"topic {topic_id}")
        reps = representative_complaints(df, topic_id, week, label, top_k)
        complaints = [
            {
                "complaint_id": int(c["complaint_id"]),
                "product": str(c["product"]),
                "narrative": str(c[config.TEXT_COLUMN]),
            }
            for _, c in reps.iterrows()
        ]
        topics_out.append(
            {
                "topic": topic_id,
                "label": label,
                "count": int(spike["count"]),
                "z_score": float(spike["z_score"]),
                "complaints": complaints,
            }
        )

    return {"week": week_str, "topics": topics_out}


def all_valid_ids(evidence: dict) -> set[int]:
    return {int(c["complaint_id"]) for topic in evidence["topics"] for c in topic["complaints"]}


if __name__ == "__main__":
    data = load_topics()
    topic_labels = build_topic_labels(data)
    spike_table = find_spikes(data)

    print("Total topic-week spikes:", len(spike_table))
    print("\nWeeks with the most spiking topics:")
    counts = spike_table.groupby("week").size().sort_values(ascending=False)
    for week_value, n in counts.head(8).items():
        print(f"  {week_value.date()}  ->  {n} spiking topics")

    pack = build_evidence(data, counts.index[0], topic_labels, spike_table)
    print(f"\nEvidence pack for {pack['week']}: {len(pack['topics'])} topics")
    for t in pack["topics"]:
        print(
            f"  topic {t['topic']:>2}  z={t['z_score']:>6}  n={t['count']:>4}  "
            f"| {t['label']}  | {len(t['complaints'])} complaints"
        )
