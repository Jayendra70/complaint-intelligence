"""
Phase 5 - Retrieval for the grounded weekly brief.

What this file does, in plain words:
  1. Loads the topic-tagged complaints from Phase 4.
  2. Rebuilds a short keyword label for each topic (so the brief can name it).
  3. Finds which topics "spiked" in a given week (same rule as Phase 4).
  4. For each spiking topic, pulls the top-k most representative complaints,
     each with its real complaint_id so the brief can cite it.

No transformer or LLM is used here. Everything is plain pandas + scikit-learn,
so it runs fast on a laptop CPU.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


# ----------------------------------------------------------------------
# 0. Settings (all in one place so they are easy to find and change)
# ----------------------------------------------------------------------

TOPICS_PARQUET = "data/processed/complaint_topics.parquet"

# Spike rule - copied exactly from the Phase 4 report so the numbers match.
TRAILING_WEEKS = 6      # look back up to 6 weeks
MIN_PRIOR_WEEKS = 3     # but only score a week once we have >= 3 weeks of history
Z_THRESHOLD = 2.0       # "spike" = >= 2 std above the trailing average
MIN_COUNT = 15          # and at least 15 complaints that week

OUTLIER_TOPIC = -1      # BERTopic's "no clear topic" bucket - always ignored

# The narratives are full of XXXX PII masks and dollar "00" fragments.
# We drop those so the topic labels read like real themes.
PII_STOPWORDS = ["xxxx", "xx", "xxxx xxxx", "xx xx", "xxxx xx", "xx xxxx", "00"]

DEFAULT_TOP_K = 5       # how many representative complaints per topic


# ----------------------------------------------------------------------
# 1. Load the data and add a "week" column
# ----------------------------------------------------------------------

def load_topics(path=TOPICS_PARQUET):
    """Read the Phase 4 parquet and add a Monday-dated 'week' column."""
    df = pd.read_parquet(path)

    # Drop the timezone, then snap each date back to the Monday of its week.
    naive_date = df["date_received"].dt.tz_localize(None)
    monday = naive_date - pd.to_timedelta(naive_date.dt.weekday, unit="D")
    df["week"] = monday.dt.normalize()
    return df


# ----------------------------------------------------------------------
# 2. Build a short keyword label for every topic
# ----------------------------------------------------------------------

def build_topic_labels(df, n_words=4):
    """
    Give each topic a short label like 'wells, wells fargo, fargo, account'.

    Method (a simple class-based TF-IDF, same idea as BERTopic's c-TF-IDF):
      - Join all narratives of a topic into one big document.
      - Run TF-IDF across those per-topic documents.
      - Take the highest-scoring words for each topic.
    """
    real = df[df["topic"] != OUTLIER_TOPIC]

    # Step 2a. One document per topic.
    joined = real.groupby("topic")["consumer_complaint_narrative"].apply(
        lambda s: " ".join(s.astype(str))
    )
    topic_ids = list(joined.index)

    # Step 2b. TF-IDF over those documents.
    stop_words = list(ENGLISH_STOP_WORDS) + PII_STOPWORDS
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=3,
    )
    matrix = vectorizer.fit_transform(joined.values)
    terms = np.array(vectorizer.get_feature_names_out())

    # Step 2c. Top words per topic -> label string.
    labels = {}
    for row_index, topic_id in enumerate(topic_ids):
        scores = matrix[row_index].toarray().ravel()
        top_positions = scores.argsort()[::-1][:n_words]
        labels[int(topic_id)] = ", ".join(terms[top_positions])
    return labels


# ----------------------------------------------------------------------
# 3. Find spiking topic-weeks (the Phase 4 rule)
# ----------------------------------------------------------------------

def find_spikes(df):
    """Return a DataFrame of every spiking topic-week: topic, week, count, z."""
    real = df[df["topic"] != OUTLIER_TOPIC]
    weekly = real.groupby(["topic", "week"]).size().reset_index(name="count")

    rows = []
    for topic_id, sub in weekly.groupby("topic"):
        sub = sub.sort_values("week").reset_index(drop=True)
        for i in range(len(sub)):
            # trailing window = up to 6 weeks before this one
            prior = sub["count"].iloc[max(0, i - TRAILING_WEEKS):i]
            if len(prior) < MIN_PRIOR_WEEKS:
                continue
            mean = prior.mean()
            std = prior.std()
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


def spikes_in_week(spikes, week):
    """Filter the spike table down to a single week (a 'YYYY-MM-DD' string or date)."""
    week = pd.Timestamp(week).normalize()
    picked = spikes[spikes["week"] == week]
    return picked.sort_values("z_score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 4. Pull representative complaints for a spiking topic
# ----------------------------------------------------------------------

def representative_complaints(df, topic_id, week, label, top_k=DEFAULT_TOP_K):
    """
    Pick the top_k complaints for this topic-and-week that best match the
    topic's keywords.

    We have no stored sentence embeddings, so we use a simple, honest score:
    how many of the topic's label keywords appear in the narrative (longer,
    on-topic complaints float to the top). Ties are broken by narrative length.
    """
    week = pd.Timestamp(week).normalize()
    keywords = [w.strip() for w in label.split(",")]

    # All complaints of this topic in this week.
    mask = (df["topic"] == topic_id) & (df["week"] == week)
    cohort = df[mask].copy()
    if cohort.empty:
        return cohort

    # Score = number of label keywords found in the (lower-cased) narrative.
    lowered = cohort["consumer_complaint_narrative"].astype(str).str.lower()
    score = np.zeros(len(cohort), dtype=int)
    for word in keywords:
        score = score + lowered.str.contains(word, regex=False).astype(int).values
    cohort["match_score"] = score
    cohort["length"] = lowered.str.len().values

    cohort = cohort.sort_values(["match_score", "length"], ascending=False)
    return cohort.head(top_k)


# ----------------------------------------------------------------------
# 5. Assemble the full evidence pack the brief generator will use
# ----------------------------------------------------------------------

def build_evidence(df, week, labels, spikes, top_k=DEFAULT_TOP_K):
    """
    Bundle everything the LLM needs for one week into a plain dictionary:

    {
      "week": "2023-01-16",
      "topics": [
        {"topic": 9, "label": "...", "count": 146, "z_score": 16.66,
         "complaints": [{"complaint_id": ..., "product": ..., "narrative": ...}, ...]},
        ...
      ]
    }
    """
    week_str = pd.Timestamp(week).normalize().strftime("%Y-%m-%d")
    week_spikes = spikes_in_week(spikes, week)

    topics_out = []
    for _, spike in week_spikes.iterrows():
        topic_id = int(spike["topic"])
        label = labels.get(topic_id, f"topic {topic_id}")
        reps = representative_complaints(df, topic_id, week, label, top_k)

        complaints = []
        for _, c in reps.iterrows():
            complaints.append(
                {
                    "complaint_id": int(c["complaint_id"]),
                    "product": str(c["product"]),
                    "narrative": str(c["consumer_complaint_narrative"]),
                }
            )

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


def all_valid_ids(evidence):
    """Set of every complaint_id that appears in an evidence pack (used by the audit)."""
    ids = set()
    for topic in evidence["topics"]:
        for c in topic["complaints"]:
            ids.add(int(c["complaint_id"]))
    return ids


# ----------------------------------------------------------------------
# 6. Quick manual check:  python -m src.brief.retrieval
# ----------------------------------------------------------------------

if __name__ == "__main__":
    data = load_topics()
    topic_labels = build_topic_labels(data)
    spike_table = find_spikes(data)

    print("Total topic-week spikes:", len(spike_table))
    print()
    print("Weeks with the most spiking topics:")
    counts = spike_table.groupby("week").size().sort_values(ascending=False)
    for week_value, n in counts.head(8).items():
        print(f"  {week_value.date()}  ->  {n} spiking topics")

    # Show the evidence pack for the busiest week.
    busiest_week = counts.index[0]
    pack = build_evidence(data, busiest_week, topic_labels, spike_table)
    print()
    print(f"Evidence pack for {pack['week']}: {len(pack['topics'])} topics")
    for t in pack["topics"]:
        print(f"  topic {t['topic']:>2}  z={t['z_score']:>6}  n={t['count']:>4}  "
              f"| {t['label']}  | {len(t['complaints'])} complaints")
