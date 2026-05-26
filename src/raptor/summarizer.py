"""LLM-based cluster summarization for RAPTOR.

Reuses the OpenAI-compatible client setup from `AnswerGenerator`, so by default
this hits the local Ollama server. Every summary is cached on disk keyed by
the exact passages + model + prompt version.
"""

from __future__ import annotations

from openai import OpenAI

from ..config import LLMConfig, LLM_PRESETS
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
        if llm_config is None:
            llm_config = LLM_PRESETS["ollama"]
        self.llm_config = llm_config
        self.client = OpenAI(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
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
        response = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        self.call_count += 1
        summary = (response.choices[0].message.content or "").strip()
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
