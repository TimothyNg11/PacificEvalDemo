"""Tests for RetrievalConfig.from_name (fail-fast config-name parsing),
the .env loader, and LLM-provider auto-detection."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.config import (
    ChunkingStrategy,
    RetrievalConfig,
    SearchStrategy,
    _load_dotenv,
    detect_llm_provider,
)


def test_from_name_parses_valid_config():
    cfg = RetrievalConfig.from_name("fixed_256__vector__k3")
    assert cfg.chunking == ChunkingStrategy.FIXED_256
    assert cfg.search == SearchStrategy.VECTOR
    assert cfg.top_k == 3


def test_from_name_parses_valid_raptor_config():
    cfg = RetrievalConfig.from_name("raptor_100__raptor_tree__k5")
    assert cfg.chunking == ChunkingStrategy.RAPTOR_100
    assert cfg.search == SearchStrategy.RAPTOR_TREE
    assert cfg.top_k == 5


def test_from_name_rejects_invalid_raptor_baseline_mix():
    """RAPTOR chunking paired with a non-RAPTOR search must raise, not
    silently construct a config that will fail deep inside the runner."""
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("raptor_100__vector__k3")


def test_from_name_rejects_baseline_chunking_with_raptor_search():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__raptor_tree__k3")


def test_from_name_rejects_malformed_name():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__vector")


def test_from_name_rejects_unknown_chunking():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("nonexistent_chunking__vector__k3")


def test_from_name_rejects_unknown_search():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__nonexistent_search__k3")


def test_from_name_rejects_bad_top_k():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__vector__three")


def test_load_dotenv_sets_new_vars(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "FOO_TEST_VAR=hello\n"
        'QUOTED_VAR="quoted value"\n'
        "SINGLE_QUOTED='single value'\n"
        "NO_EQUALS_LINE_IS_IGNORED\n",
        encoding="utf-8",
    )
    for var in ("FOO_TEST_VAR", "QUOTED_VAR", "SINGLE_QUOTED"):
        monkeypatch.delenv(var, raising=False)

    _load_dotenv(str(env_file))

    assert os.environ["FOO_TEST_VAR"] == "hello"
    assert os.environ["QUOTED_VAR"] == "quoted value"
    assert os.environ["SINGLE_QUOTED"] == "single value"


def test_load_dotenv_does_not_override_existing_env_vars(tmp_path, monkeypatch):
    """A real environment variable (shell/CI) must win over .env — this is
    why the loader uses os.environ.setdefault, not direct assignment."""
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_TEST_VAR=from_dotenv\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING_TEST_VAR", "from_shell")

    _load_dotenv(str(env_file))

    assert os.environ["EXISTING_TEST_VAR"] == "from_shell"


def test_load_dotenv_missing_file_is_a_noop(tmp_path):
    _load_dotenv(str(tmp_path / "does_not_exist.env"))  # must not raise


def test_detect_llm_provider_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    assert detect_llm_provider() == "anthropic"


def test_detect_llm_provider_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    assert detect_llm_provider() == "openai"


def test_detect_llm_provider_falls_back_to_ollama(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert detect_llm_provider() == "ollama"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
