"""Build, audit and save the weekly brief for one week.

python -m src.brief.run_weekly_brief                    # busiest week, Groq
python -m src.brief.run_weekly_brief --week 2023-01-16
python -m src.brief.run_weekly_brief --offline          # template, no API call
python -m src.brief.run_weekly_brief --dry-run          # show the evidence only
"""

import argparse

from src import config
from src.brief import audit, generate, retrieval

# Groq's free tier costs nothing; set this if you move to a paid model.
USD_PER_1K_TOTAL_TOKENS = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the weekly complaint brief.")
    parser.add_argument("--week", help="Monday of the week, YYYY-MM-DD (default: busiest week)")
    parser.add_argument("--offline", action="store_true", help="use the template, no LLM call")
    parser.add_argument("--dry-run", action="store_true", help="print the evidence and stop")
    args = parser.parse_args()

    print("Loading topics and computing spikes ...")
    df = retrieval.load_topics()
    labels = retrieval.build_topic_labels(df)
    spikes = retrieval.find_spikes(df)

    week = args.week
    if week is None:
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

    if args.dry_run:
        print("\n--dry-run: evidence only, no brief generated.")
        return

    print(f"\nGenerating brief ({'offline template' if args.offline else 'Groq'}) ...")
    result = generate.generate_brief(evidence, offline=args.offline)
    audit_results = audit.audit_brief(result["text"], retrieval.all_valid_ids(evidence))

    out_path = config.REPORTS_DIR / f"weekly_brief_{evidence['week']}.md"
    with open(out_path, "w") as f:
        if result["backend"] != "offline-template":
            # The template writes its own header; give LLM output one that records
            # the model and the audit verdict.
            verdict = "PASS" if audit_results["passed"] else "FAIL"
            f.write(f"# Weekly Complaint Brief - week of {evidence['week']}\n\n")
            f.write(
                f"*Generated in LLM mode ({result['backend']}). Groundedness audit: {verdict} "
                f"({audit_results['hallucinated_count']} invented IDs, "
                f"{audit_results['unsupported_claim_lines']} of "
                f"{audit_results['total_claim_lines']} claim lines uncited).*\n\n"
            )
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
    print(
        "  tokens ...............",
        tokens,
        f"(prompt {result['usage']['prompt_tokens']}, "
        f"completion {result['usage']['completion_tokens']})",
    )
    print("  est. cost ............ $", round(cost, 6))
    print("  latency ..............", result["latency_seconds"], "s")
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
