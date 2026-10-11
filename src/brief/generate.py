"""
Phase 5 - Turn a week's evidence pack into a written weekly brief.

Two ways to make the brief:
  1. LLM mode (default) - sends the evidence to a free Groq model and asks for
     a short exec brief. Good prose. Needs a free GROQ_API_KEY (no card needed).
  2. Offline mode - builds a plain brief straight from the evidence, no API call.
     Weaker writing, but 100% grounded and free. Handy for testing and for
     running the groundedness audit without spending a token.

Either way, every factual line must cite real complaint IDs like [ID: 6382604],
and those IDs only ever come from the evidence pack - never invented.
"""

import os
import time
import textwrap


# ----------------------------------------------------------------------
# 0. Backend settings  (<<< SWAP POINT >>> - change these to use a different
#    free provider, e.g. Google Gemini's OpenAI-compatible endpoint)
# ----------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_KEY_ENV = "GROQ_API_KEY"   # name of the env var that holds the key; the
                                # actual key lives in the .env file (never committed)

# Models to try, in order. Groq retires and renames models fairly often, so
# instead of hardcoding one name (which breaks with a 404 when it is retired),
# we try each one until the account accepts it and use the first that works.
# Put the model you most want first.
GROQ_MODELS = [
    "openai/gpt-oss-120b",      # large, strong prose, currently on Groq
    "llama-3.3-70b-versatile",  # Llama 70B (was the old default)
    "llama-3.1-8b-instant",     # small + fast fallback, almost always available
]

MAX_NARRATIVE_CHARS = 700   # trim each quoted complaint so prompts stay small


# ----------------------------------------------------------------------
# 1. Build the prompt from an evidence pack
# ----------------------------------------------------------------------

def _format_evidence(evidence):
    """Turn the evidence dict into readable text for the model to work from."""
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


def build_messages(evidence):
    """Return the system + user messages for the chat model."""
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


# ----------------------------------------------------------------------
# 2. LLM call  (<<< SWAP POINT >>> - the only place that talks to a provider)
# ----------------------------------------------------------------------

def call_llm(messages):
    """
    Send the messages to Groq and return (text, usage_dict).

    Uses the OpenAI client pointed at Groq's compatible endpoint, so switching
    provider is just changing base_url / model / key above. Requires:
        pip install openai python-dotenv
    The key is read from a local .env file (GROQ_API_KEY=...), which is
    gitignored and never committed. Copy .env.example to .env and paste your
    free key from https://console.groq.com (no card needed). Setting the shell
    variable  export GROQ_API_KEY=...  also works and overrides the .env value.
    """
    from openai import OpenAI  # imported here so offline mode needs no install
    from openai import NotFoundError  # raised when a model is retired / not on the account
    from dotenv import load_dotenv  # ditto - only needed for the LLM path

    # Read GROQ_API_KEY from a .env file in the repo if it isn't already set in
    # the shell. load_dotenv() does nothing if there is no .env file, and it
    # does not overwrite a variable that is already exported.
    load_dotenv()

    api_key = os.environ.get(GROQ_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"No {GROQ_KEY_ENV} found. Copy .env.example to .env and paste a free "
            f"key from https://console.groq.com, or run:  export {GROQ_KEY_ENV}=your_key"
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    # Try each model in turn. If one is retired/unavailable Groq returns a 404
    # (NotFoundError); we catch that and fall through to the next model. Any
    # other error (bad key, network, rate limit) is a real problem, so we let it
    # raise instead of hiding it.
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
            continue

    if response is None:
        raise RuntimeError(
            "None of the models in GROQ_MODELS were available on this account: "
            f"{GROQ_MODELS}. Check the current list at https://console.groq.com/docs/models "
            "and update GROQ_MODELS in src/brief/generate.py."
        )

    text = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return text, usage, used_model


# ----------------------------------------------------------------------
# 3. Offline template brief (no API, always grounded)
# ----------------------------------------------------------------------

def offline_brief(evidence):
    """Build a simple grounded brief from the evidence with no LLM call."""
    week = evidence["week"]
    topics = evidence["topics"][:4]  # top few by z-score (already sorted)

    out = [f"# Weekly Complaint Brief - week of {week}", ""]
    out.append("*Generated in offline template mode (no LLM). Every line cites "
               "real complaint IDs.*")
    out.append("")

    # Headline
    if topics:
        top = topics[0]
        ids = ", ".join(f"[ID: {c['complaint_id']}]" for c in top["complaints"][:2])
        out.append("## Headline")
        out.append(
            f"Complaints about '{top['label']}' spiked to {top['count']} this week "
            f"(z-score {top['z_score']}), the biggest jump of the week. {ids}"
        )
        out.append("")

    # Trend
    out.append("## Trend")
    for t in topics:
        ids = ", ".join(f"[ID: {c['complaint_id']}]" for c in t["complaints"][:2])
        out.append(
            f"- '{t['label']}' rose to {t['count']} complaints "
            f"(z-score {t['z_score']}). {ids}"
        )
    out.append("")

    # Evidence
    out.append("## Evidence")
    for t in topics:
        out.append(f"**{t['label']}**")
        for c in t["complaints"][:2]:
            snippet = c["narrative"][:220].replace("\n", " ").strip()
            out.append(f"- {snippet} ... [ID: {c['complaint_id']}]")
        out.append("")

    # Recommended action
    out.append("## Recommended action")
    if topics:
        top = topics[0]
        ids = ", ".join(f"[ID: {c['complaint_id']}]" for c in top["complaints"][:3])
        out.append(
            f"Route the '{top['label']}' cases to the relevant team and check for a "
            f"common root cause this week. {ids}"
        )
    return "\n".join(out)


# ----------------------------------------------------------------------
# 4. One entry point that both the runner and tests call
# ----------------------------------------------------------------------

def generate_brief(evidence, offline=False):
    """
    Make a brief and return a result dict:
      {"text": ..., "backend": ..., "usage": ..., "latency_seconds": ...}
    """
    start = time.time()

    if offline:
        text = offline_brief(evidence)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        backend = "offline-template"
    else:
        messages = build_messages(evidence)
        text, usage, used_model = call_llm(messages)
        backend = f"groq:{used_model}"

    latency = round(time.time() - start, 2)
    return {
        "text": text,
        "backend": backend,
        "usage": usage,
        "latency_seconds": latency,
    }
