"""Groundedness audit: every claim line must cite a complaint ID that was in the evidence."""

import re
from collections.abc import Iterable

ID_PATTERN = re.compile(r"\[ID:\s*(\d+)\]")


def find_citations(text: str) -> list[int]:
    return [int(match) for match in ID_PATTERN.findall(text)]


def _is_claim_line(line: str) -> bool:
    """A claim is any non-empty line that isn't a heading or a line wrapped in * / ** ."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    # Italic notes (*...*) and bold labels (**wells fargo**) are structure, not claims.
    return not (stripped.startswith("*") and stripped.endswith("*"))


def audit_brief(text: str, valid_ids: Iterable[int]) -> dict:
    """Flag cited IDs that aren't in `valid_ids` and claim lines with no citation.

    The brief passes only if both counts are zero.
    """
    valid_ids = {int(i) for i in valid_ids}

    cited = find_citations(text)
    unique_cited = set(cited)
    hallucinated = sorted(i for i in unique_cited if i not in valid_ids)

    claim_lines = [ln for ln in text.splitlines() if _is_claim_line(ln)]
    uncited_claims = [ln.strip() for ln in claim_lines if not ID_PATTERN.search(ln)]

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


def print_audit(results: dict) -> None:
    print("Groundedness audit")
    print(
        "  citations found ......",
        results["total_citations"],
        f"({results['unique_ids_cited']} unique IDs)",
    )
    print(
        "  hallucinated IDs .....",
        results["hallucinated_count"],
        results["hallucinated_ids"] or "",
    )
    print("  claim lines ..........", results["total_claim_lines"])
    print(
        "  unsupported claims ...",
        results["unsupported_claim_lines"],
        f"(rate {results['unsupported_claim_rate']})",
    )
    if results["uncited_examples"]:
        print("  examples missing a citation:")
        for ex in results["uncited_examples"]:
            print("    -", ex[:90])
    print("  RESULT ...............", "PASS" if results["passed"] else "FAIL")
