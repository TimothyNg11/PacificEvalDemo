"""LLM answer generation.

Supports two families of provider, both reached through `resolve_llm_client`:
- OpenAI-compatible (Ollama, OpenAI) via the `openai` SDK.
- Anthropic (Claude) via the official `anthropic` SDK — a separate code
  path, not an openai-SDK shim pointed at an Anthropic base_url.

Both are wrapped behind one uniform `chat(system, user, max_tokens)`
interface so callers (`AnswerGenerator`, `raptor.summarizer.Summarizer`)
don't need to know which provider they're talking to.
"""

import time
from dataclasses import dataclass

import tiktoken
from openai import OpenAI

from .chunkers import Chunk
from .config import LLMConfig, LLM_PRESETS


@dataclass
class GenerationResult:
    answer: str
    context_tokens: int
    generation_latency_ms: float


_tokenizer = tiktoken.get_encoding("cl100k_base")

# Transient-failure retries around each chat call, on top of the SDKs' own
# internal retries. Long unattended runs (a tree build makes hundreds of
# sequential calls over ~an hour) must survive a network blip that outlasts
# the SDK's ~2 quick retries: back off 10s/20s/40s before giving up.
_RETRY_ATTEMPTS = 4
_RETRY_BASE_SLEEP_S = 10.0


def _chat_with_retries(call, transient_errors: tuple):
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return call()
        except transient_errors:
            if attempt == _RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_BASE_SLEEP_S * 2 ** attempt)


class _OpenAICompatChat:
    """Uniform chat() interface wrapping an OpenAI-compatible client
    (Ollama or OpenAI)."""

    def __init__(self, model: str, client: OpenAI):
        self.model = model
        self._client = client

    def chat(self, system: str, user: str, max_tokens: int, temperature: float = 0.0) -> str:
        import openai

        def call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        response = _chat_with_retries(
            call,
            (openai.APIConnectionError, openai.RateLimitError,
             openai.InternalServerError),
        )
        return response.choices[0].message.content or ""


class _AnthropicChat:
    """Uniform chat() interface wrapping the official `anthropic` SDK.

    Per Anthropic API guidance: a plain `messages.create()` call with no
    temperature/top_p/thinking params (`temperature` is accepted here only
    to match `_OpenAICompatChat`'s signature — it is intentionally never
    forwarded). The SDK's default retries (max_retries=2) handle quick
    429/5xx blips; `_chat_with_retries` adds backoff for outages that
    outlast them.
    """

    def __init__(self, model: str, client):
        self.model = model
        self._client = client

    def chat(self, system: str, user: str, max_tokens: int, temperature: float = 0.0) -> str:
        import anthropic

        def call():
            return self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )

        response = _chat_with_retries(
            call,
            (anthropic.APIConnectionError, anthropic.RateLimitError,
             anthropic.InternalServerError),
        )
        return "".join(block.text for block in response.content if block.type == "text")


def resolve_llm_client(llm_config: LLMConfig | None):
    """Resolve an `LLMConfig` (defaulting to the local Ollama preset) and
    build the matching client, wrapped behind a uniform `chat(system, user,
    max_tokens) -> str` interface. Shared by `AnswerGenerator` and
    `raptor.summarizer.Summarizer` so there's one place that knows how to
    turn an `LLMConfig` into a live, provider-appropriate client.
    """
    if llm_config is None:
        llm_config = LLM_PRESETS["ollama"]

    if llm_config.provider_name == "anthropic":
        import anthropic

        # anthropic.Anthropic() picks up ANTHROPIC_API_KEY from the
        # environment on its own — no api_key argument passed here.
        client = anthropic.Anthropic()
        return llm_config, _AnthropicChat(llm_config.model, client)

    client = OpenAI(base_url=llm_config.base_url, api_key=llm_config.api_key)
    return llm_config, _OpenAICompatChat(llm_config.model, client)


class AnswerGenerator:
    """Generates answers from retrieved context chunks using an LLM."""

    def __init__(self, llm_config: LLMConfig | None = None):
        self.llm_config, self.client = resolve_llm_client(llm_config)

    def generate(self, question: str, context_chunks: list[Chunk]) -> GenerationResult:
        # Assemble context
        context = "\n---\n".join(chunk.text for chunk in context_chunks)
        context_tokens = len(_tokenizer.encode(context))

        system_prompt = (
            "You are a helpful assistant. Answer the question based ONLY on "
            "the provided context. If the context does not contain enough "
            "information to fully answer the question, say so explicitly. "
            "Do not make up information."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        start = time.perf_counter()
        answer = self.client.chat(
            system=system_prompt, user=user_prompt, max_tokens=1024, temperature=0.1,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return GenerationResult(
            answer=answer,
            context_tokens=context_tokens,
            generation_latency_ms=elapsed_ms,
        )
