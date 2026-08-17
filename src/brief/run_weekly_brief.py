"""
Phase 5 - Run the whole weekly-brief pipeline for one week.

Steps:
  1. Load Phase 4 topics, build labels, find spikes.
  2. Build the evidence pack for the chosen week.
  3. Generate the brief (Groq by default, or --offline for no API).
  4. Audit it for groundedness.
  5. Save the brief to reports/ and print cost/latency + audit summary.

Usage:
  export GROQ_API_KEY=your_free_key
  python -m src.brief.run_weekly_brief                 # busiest week, Groq
  python -m src.brief.run_weekly_brief --week 2023-01-16
  python -m src.brief.run_weekly_brief --offline       # no API call
  python -m src.brief.run_weekly_brief --dry-run       # show evidence only
"""

import sys

from src.brief import retrieval
from src.brief import generate
from src.brief import audit


# Groq's free tier is genuinely free (no card), so cost is $0. We still print
# tokens so you can see the size. If you ever swap to a paid model, set this.
USD_PER_1K_TOTAL_TOKENS = 0.0


def _get_arg(name, default=None):
    """Tiny helper to read '--name value' from the command line."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    offline = "--offline" in sys.argv
    dry_run = "--dry-run" in sys.argv

    # 1. Load + prepare.
    print("Loading topics and computing spikes ...")
    df = retrieval.load_topics()
    labels = retrieval.build_topic_labels(df)
    spikes = retrieval.find_spikes(df)

    # 2. Choose the week.
    week = _get_arg("--week")
    if week is None:
        # default: the week with the most spiking topics
        week = spikes.groupby("week").size().sort_values(ascending=False).index[0]
        week = week.strftime("%Y-%m-%d")
    print(f"Week: {week}")

    evidence = retrieval.build_evidence(df, week, labels, spikes)
    if not evidence["topics"]:
        print("No spikes in that week. Try --week with a spiking week.")
        return

    print(f"Spiking topics this week: {len(evidence['topics'])}")
    for t in evidence["topics"]:
        print(f"  topic {t['topic']:>2}  z={t['z_score']:>6}  n={t['count']:>4}  | {t['label']}")

    if dry_run:
        print("\n--dry-run: evidence only, no brief generated.")
        return

    # 3. Generate.
    print(f"\nGenerating brief ({'offline template' if offline else 'Groq'}) ...")
    result = generate.generate_brief(evidence, offline=offline)

    # 4. Audit.
    valid_ids = retrieval.all_valid_ids(evidence)
    audit_results = audit.audit_brief(result["text"], valid_ids)

    # 5. Save + report.
    out_path = f"reports/weekly_brief_{evidence['week']}.md"
    with open(out_path, "w") as f:
        f.write(result["text"])
        f.write("\n")

    print("\n" + "=" * 60)
    print(result["text"])
    print("=" * 60 + "\n")

    audit.print_audit(audit_results)

    tokens = result["usage"]["total_tokens"]
    cost = tokens / 1000.0 * USD_PER_1K_TOTAL_TOKENS
    print("\nCost / latency")
    print("  backend ..............", result["backend"])
    print("  tokens ...............", tokens,
          f"(prompt {result['usage']['prompt_tokens']}, "
          f"completion {result['usage']['completion_tokens']})")
    print("  est. cost ............ $", round(cost, 6))
    print("  latency ..............", result["latency_seconds"], "s")
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
