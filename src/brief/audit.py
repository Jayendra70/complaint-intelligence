"""
Phase 5 - Groundedness audit for a weekly brief.

The whole point of the brief is that it does NOT make things up. This file
checks that claim by claim:

  1. Find every [ID: <number>] citation in the brief text.
  2. Check each cited ID actually exists in the evidence pack (an ID the model
     was allowed to use). Any ID that is not in the evidence is a hallucination.
  3. Find factual sentences that carry NO citation at all (unsupported claims).
  4. Report the unsupported-claim rate. Target = 0.

A tiny regex is used only to pull the "[ID: 12345]" tokens out of the text -
that is the one job regex is genuinely the simplest tool for.
"""

import re


# Matches "[ID: 6382604]" and captures the number. Spaces are optional.
_ID_PATTERN = re.compile(r"\[ID:\s*(\d+)\]")

# Section headings whose lines are just structure, not factual claims.
_HEADING_PREFIXES = ("#", "*", "**", "-")


def find_citations(text):
    """Return the list of complaint IDs (ints) cited anywhere in the text."""
    return [int(match) for match in _ID_PATTERN.findall(text)]


def _is_claim_line(line):
    """A 'claim' is a non-empty content line that is not a bare heading/label."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):        # markdown heading
        return False
    if stripped.startswith("*") and stripped.endswith("*"):  # italic note
        return False
    # A bold-only label line like "**wells fargo**" is a label, not a claim.
    if stripped.startswith("**") and stripped.endswith("**") and stripped.count("*") == 4:
        return False
    return True


def audit_brief(text, valid_ids):
    """
    Run the full audit and return a plain results dictionary.

    valid_ids: the set of complaint IDs the brief was allowed to cite
               (use retrieval.all_valid_ids(evidence)).
    """
    valid_ids = set(int(i) for i in valid_ids)

    # 1. Citations overall.
    cited = find_citations(text)
    unique_cited = set(cited)

    # 2. Hallucinated IDs = cited but not in the allowed evidence.
    hallucinated = sorted(i for i in unique_cited if i not in valid_ids)

    # 3. Claim lines and which of them lack any citation.
    claim_lines = [ln for ln in text.splitlines() if _is_claim_line(ln)]
    uncited_claims = [ln.strip() for ln in claim_lines if not _ID_PATTERN.search(ln)]

    total_claims = len(claim_lines)
    unsupported = len(uncited_claims)
    unsupported_rate = round(unsupported / total_claims, 3) if total_claims else 0.0

    return {
        "total_citations": len(cited),
        "unique_ids_cited": len(unique_cited),
        "hallucinated_ids": hallucinated,
        "hallucinated_count": len(hallucinated),
        "total_claim_lines": total_claims,
        "unsupported_claim_lines": unsupported,
        "unsupported_claim_rate": unsupported_rate,
        "uncited_examples": uncited_claims[:5],
        "passed": len(hallucinated) == 0 and unsupported == 0,
    }


def print_audit(results):
    """Pretty-print the audit results for the terminal."""
    print("Groundedness audit")
    print("  citations found ......", results["total_citations"],
          f"({results['unique_ids_cited']} unique IDs)")
    print("  hallucinated IDs .....", results["hallucinated_count"],
          results["hallucinated_ids"] if results["hallucinated_ids"] else "")
    print("  claim lines ..........", results["total_claim_lines"])
    print("  unsupported claims ...", results["unsupported_claim_lines"],
          f"(rate {results['unsupported_claim_rate']})")
    if results["uncited_examples"]:
        print("  examples missing a citation:")
        for ex in results["uncited_examples"]:
            print("    -", ex[:90])
    print("  RESULT ...............", "PASS" if results["passed"] else "FAIL")
