"""LLM-based cluster summarization for RAPTOR.

Reuses the client bootstrap from `generator.resolve_llm_client` (the same
helper `AnswerGenerator` uses), so this works on any configured provider —
Ollama, OpenAI, or Anthropic — by default hitting the local Ollama server.
Every summary is cached on disk keyed by the exact passages + model +
prompt version (model string alone is enough to distinguish providers
since presets use distinct model names).
"""

from __future__ import annotations

from ..config import LLMConfig
from ..generator import resolve_llm_client
from .cache import SummaryCache, summary_key
from .prompts import PROMPT_VERSION, SUMMARIZE_SYSTEM, build_summarize_user_prompt


class Summarizer:
    """Summarize a cluster of passages into a single dense paragraph."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        cache: SummaryCache | None = None,
        max_input_chars: int = 24_000,
    ):
        self.llm_config, self.client = resolve_llm_client(llm_config)
        self.cache = cache if cache is not None else SummaryCache()
        self.max_input_chars = max_input_chars
        self.call_count = 0
        self.cache_hits = 0

    def summarize(self, passages: list[str]) -> str:
        if not passages:
            return ""

        truncated = self._truncate(passages)
        key = summary_key(truncated, self.llm_config.model, PROMPT_VERSION)

        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        user_prompt = build_summarize_user_prompt(truncated)
        # SUMMARIZE_SYSTEM asks for ~250 tokens; 512 leaves headroom so the
        # model isn't cut off mid-sentence right at its own target length.
        raw_summary = self.client.chat(
            system=SUMMARIZE_SYSTEM, user=user_prompt, max_tokens=512, temperature=0.0,
        )
        self.call_count += 1
        summary = raw_summary.strip()
        self.cache.put(key, summary)
        return summary

    def _truncate(self, passages: list[str]) -> list[str]:
        """Truncate the total passage payload to fit small-context local LLMs.

        Keeps as many passages as possible, then truncates the last one.
        """
        budget = self.max_input_chars
        kept: list[str] = []
        for p in passages:
            if budget <= 0:
                break
            if len(p) <= budget:
                kept.append(p)
                budget -= len(p)
            else:
                kept.append(p[:budget])
                budget = 0
                break
        return kept
