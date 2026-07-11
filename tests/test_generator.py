"""Tests for the LLM client bootstrap (resolve_llm_client) and
AnswerGenerator, covering both the OpenAI-compatible and Anthropic chat
paths. Network-free: both provider SDKs are stubbed.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunkers import Chunk
from src.config import LLM_PRESETS
from src.generator import AnswerGenerator, _OpenAICompatChat


def _chunks():
    return [
        Chunk(text="Some context.", source_file="a.md", chunk_index=0, chunking_strategy="test"),
    ]


def test_answer_generator_openai_compat_chat_path():
    """_OpenAICompatChat wraps an OpenAI-style client and passes through
    max_tokens/temperature as AnswerGenerator.generate() sets them."""
    calls = {}

    class _FakeMessage:
        content = "Stubbed answer from OpenAI-compat."

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return types.SimpleNamespace(choices=[_FakeChoice()])

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    chat = _OpenAICompatChat(model="gpt-4o-mini", client=_FakeClient())
    result = chat.chat(system="sys", user="usr", max_tokens=1024, temperature=0.1)

    assert result == "Stubbed answer from OpenAI-compat."
    assert calls["kwargs"]["model"] == "gpt-4o-mini"
    assert calls["kwargs"]["max_tokens"] == 1024
    assert calls["kwargs"]["temperature"] == 0.1
    assert calls["kwargs"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_answer_generator_uses_anthropic_chat_path(monkeypatch):
    """AnswerGenerator.generate() on the anthropic preset must go through
    the official anthropic SDK's messages.create(), with no api_key passed
    to the client constructor (the SDK reads ANTHROPIC_API_KEY itself) and
    no temperature/top_p/thinking sent to messages.create()."""
    import anthropic

    calls = {}

    class _FakeMessages:
        def create(self, **kwargs):
            calls["create_kwargs"] = kwargs
            block = types.SimpleNamespace(type="text", text="Stubbed answer from Claude.")
            other_block = types.SimpleNamespace(type="other", text="ignored")
            return types.SimpleNamespace(content=[other_block, block])

    class _FakeAnthropicClient:
        def __init__(self, *args, **kwargs):
            calls["init_args"] = args
            calls["init_kwargs"] = kwargs
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropicClient)

    gen = AnswerGenerator(llm_config=LLM_PRESETS["anthropic"])
    result = gen.generate("What is in the context?", _chunks())

    assert result.answer == "Stubbed answer from Claude."
    assert result.context_tokens > 0

    # No api_key argument — anthropic.Anthropic() picks up ANTHROPIC_API_KEY
    # from the environment on its own.
    assert calls["init_args"] == ()
    assert "api_key" not in calls["init_kwargs"]

    create_kwargs = calls["create_kwargs"]
    assert create_kwargs["model"] == "claude-haiku-4-5"
    assert create_kwargs["max_tokens"] == 1024
    assert create_kwargs["system"] == (
        "You are a helpful assistant. Answer the question based ONLY on "
        "the provided context. If the context does not contain enough "
        "information to fully answer the question, say so explicitly. "
        "Do not make up information."
    )
    assert create_kwargs["messages"] == [
        {"role": "user", "content": create_kwargs["messages"][0]["content"]}
    ]
    # No temperature/top_p/thinking params sent to Anthropic.
    assert "temperature" not in create_kwargs
    assert "top_p" not in create_kwargs
    assert "thinking" not in create_kwargs


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
