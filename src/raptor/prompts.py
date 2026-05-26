"""LLM prompt templates for RAPTOR summarization.

PROMPT_VERSION is part of the cache key — bump it on any prompt change.
"""

PROMPT_VERSION = "v1"

SUMMARIZE_SYSTEM = (
    "You are summarizing a set of related text passages from a corpus. "
    "Produce a concise, information-dense summary that preserves named "
    "entities, numbers, dates, technical terms, and any factual claims. "
    "Do not introduce information not present in the passages. "
    "Do not add commentary, preamble, or meta-text — output only the summary. "
    "Limit to about 250 tokens."
)


def build_summarize_user_prompt(passages: list[str]) -> str:
    joined = "\n\n---\n\n".join(passages)
    return (
        "Summarize the following passages:\n\n"
        f"{joined}\n\n"
        "Summary:"
    )
