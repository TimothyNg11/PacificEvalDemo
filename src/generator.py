"""LLM answer generation using OpenAI-compatible API (Ollama or OpenAI)."""

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


def resolve_llm_client(llm_config: LLMConfig | None) -> tuple[LLMConfig, OpenAI]:
    """Resolve an `LLMConfig` (defaulting to the local Ollama preset) and
    build the matching OpenAI-compatible client. Shared by `AnswerGenerator`
    and `raptor.summarizer.Summarizer` so there's one place that knows how
    to turn an `LLMConfig` into a live client.
    """
    if llm_config is None:
        llm_config = LLM_PRESETS["ollama"]
    client = OpenAI(base_url=llm_config.base_url, api_key=llm_config.api_key)
    return llm_config, client


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
        response = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        answer = response.choices[0].message.content or ""

        return GenerationResult(
            answer=answer,
            context_tokens=context_tokens,
            generation_latency_ms=elapsed_ms,
        )
