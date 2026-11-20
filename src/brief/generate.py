"""Write a weekly brief from an evidence pack, with an LLM (Groq) or an offline template.

Every factual line cites complaint IDs as [ID: 6382604], taken only from the evidence.
"""

import os
import time

# Groq exposes an OpenAI-compatible API; point these at another provider to switch.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_KEY_ENV = "GROQ_API_KEY"

# Groq retires models often, so try these in order and use the first one available.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

MAX_NARRATIVE_CHARS = 700


def _format_evidence(evidence: dict) -> str:
    lines = [f"WEEK: {evidence['week']}", ""]
    for topic in evidence["topics"]:
        lines.append(
            f"TOPIC {topic['topic']} - keywords: {topic['label']} "
            f"(count={topic['count']}, z-score={topic['z_score']})"
        )
        for c in topic["complaints"]:
            snippet = c["narrative"][:MAX_NARRATIVE_CHARS].replace("\n", " ")
            lines.append(f"  [ID: {c['complaint_id']}] ({c['product']}) {snippet}")
        lines.append("")
    return "\n".join(lines)


def build_messages(evidence: dict) -> list[dict]:
    system = (
        "You are a data analyst writing a short weekly complaint brief for a "
        "bank's operations team. You must be factual and grounded. Every claim "
        "must be supported by at least one complaint ID from the evidence, "
        "written exactly as [ID: <number>]. Only use IDs that appear in the "
        "evidence. Never invent an ID or a fact. If the evidence is thin, say so."
    )
    user = (
        "Write a weekly brief from the evidence below.\n\n"
        "Format (use these exact section headings):\n"
        "## Headline\n"
        "## Trend\n"
        "## Evidence\n"
        "## Recommended action\n\n"
        "Rules:\n"
        "- Cover the 3-4 most important spiking topics, not all of them.\n"
        "- Every line under every heading, including the headline and each "
        "recommended action, must end with one or more [ID: ...] citations drawn "
        "from the evidence. For an action, cite the complaints that motivate it.\n"
        "- Write each heading exactly as shown, on its own line, without bold.\n"
        "- Do not add any closing sentence after the last recommended action.\n"
        "- Keep it under 300 words. Plain business English.\n\n"
        "EVIDENCE:\n"
        f"{_format_evidence(evidence)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(messages: list[dict]) -> tuple[str, dict, str]:
    """Send the chat to Groq. Returns (text, token usage, model used)."""
    # Imported lazily so offline mode and the tests don't need these packages.
    from dotenv import load_dotenv
    from openai import NotFoundError, OpenAI

    load_dotenv()
    api_key = os.environ.get(GROQ_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"No {GROQ_KEY_ENV} found. Copy .env.example to .env and paste a free "
            f"key from https://console.groq.com, or run:  export {GROQ_KEY_ENV}=your_key"
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    # A retired model returns 404 (NotFoundError), so move on to the next one.
    # Anything else (bad key, rate limit, network) should surface.
    response = None
    used_model = None
    for model in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            used_model = model
            break
        except NotFoundError:
            print(f"  model '{model}' unavailable, trying the next one ...")

    if response is None:
        raise RuntimeError(
            "None of the models in GROQ_MODELS were available on this account: "
            f"{GROQ_MODELS}. Check the current list at https://console.groq.com/docs/models "
            "and update GROQ_MODELS in src/brief/generate.py."
        )

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return response.choices[0].message.content, usage, used_model


def offline_brief(evidence: dict) -> str:
    """Template brief built straight from the evidence, no API call."""
    week = evidence["week"]
    topics = evidence["topics"][:4]

    out = [f"# Weekly Complaint Brief - week of {week}", ""]
    out.append(
        "*Generated in offline template mode (no LLM). Every line cites real complaint IDs.*"
    )
    out.append("")

    if topics:
        top = topics[0]
        ids = ", ".join(f"[ID: {c['complaint_id']}]" for c in top["complaints"][:2])
        out.append("## Headline")
        out.append(
            f"Complaints about '{top['label']}' spiked to {top['count']} this week "
            f"(z-score {top['z_score']}), the biggest jump of the week. {ids}"
        )
        out.append("")

    out.append("## Trend")
    for t in topics:
        ids = ", ".join(f"[ID: {c['complaint_id']}]" for c in t["complaints"][:2])
        out.append(
            f"- '{t['label']}' rose to {t['count']} complaints (z-score {t['z_score']}). {ids}"
        )
    out.append("")

    out.append("## Evidence")
    for t in topics:
        out.append(f"**{t['label']}**")
        for c in t["complaints"][:2]:
            snippet = c["narrative"][:220].replace("\n", " ").strip()
            out.append(f"- {snippet} ... [ID: {c['complaint_id']}]")
        out.append("")

    out.append("## Recommended action")
    if topics:
        top = topics[0]
        ids = ", ".join(f"[ID: {c['complaint_id']}]" for c in top["complaints"][:3])
        out.append(
            f"Route the '{top['label']}' cases to the relevant team and check for a "
            f"common root cause this week. {ids}"
        )
    return "\n".join(out)


def generate_brief(evidence: dict, offline: bool = False) -> dict:
    """Return {"text", "backend", "usage", "latency_seconds"} for one evidence pack."""
    start = time.time()

    if offline:
        text = offline_brief(evidence)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        backend = "offline-template"
    else:
        text, usage, used_model = call_llm(build_messages(evidence))
        backend = f"groq:{used_model}"

    return {
        "text": text,
        "backend": backend,
        "usage": usage,
        "latency_seconds": round(time.time() - start, 2),
    }
