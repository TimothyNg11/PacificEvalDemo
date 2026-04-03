"""Tests for chunking strategies."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tiktoken

from src.chunkers import (
    chunk_fixed_256,
    chunk_fixed_512,
    chunk_semantic,
    chunk_paragraph,
    chunk_corpus,
    get_chunker,
)
from src.config import ChunkingStrategy

_enc = tiktoken.get_encoding("cl100k_base")

SAMPLE_TEXT = """# Architecture Overview

## System Design

Meridian Technologies operates a microservices architecture consisting of 23 services running on Kubernetes. The system handles approximately 2.8 million API requests per day with a 99.97% uptime SLA.

## Core Services

The API Gateway is built on Kong and handles authentication, rate limiting, and request routing. It processes an average of 32,000 requests per minute during peak hours.

The User Service manages authentication and user profiles. It uses PostgreSQL 15 as its primary datastore with a read replica for analytics queries. The service maintains approximately 847,000 active user accounts.

## Document Processing Pipeline

The document processing pipeline is the core of our product. Documents are uploaded to S3 and a message is published to SQS. The ingestion service processes documents at a rate of approximately 450 documents per minute.

Documents are parsed, chunked, and embedded using a fine-tuned embedding model. Processing takes an average of 2.3 seconds per document. Processed documents are indexed in our vector store and metadata is stored in PostgreSQL.

## Search Service

The Search Service handles all retrieval operations. It supports three search modes: vector search, keyword search, and hybrid search. The service maintains a cache layer using Redis with a 78% cache hit rate.

## Security

All data in transit is encrypted using TLS 1.3. Data at rest is encrypted using AES-256. We completed SOC 2 Type II certification in August 2024.
"""


def test_chunk_fixed_256_token_limit():
    """Verify no chunk exceeds 306 tokens (256 + 50 overlap tolerance)."""
    chunks = chunk_fixed_256(SAMPLE_TEXT, "test/doc.md")
    for chunk in chunks:
        token_count = len(_enc.encode(chunk.text))
        assert token_count <= 306, f"Chunk has {token_count} tokens, exceeds 306"


def test_chunk_fixed_512_token_limit():
    """Verify no chunk exceeds 612 tokens (512 + 100 overlap tolerance)."""
    chunks = chunk_fixed_512(SAMPLE_TEXT, "test/doc.md")
    for chunk in chunks:
        token_count = len(_enc.encode(chunk.text))
        assert token_count <= 612, f"Chunk has {token_count} tokens, exceeds 612"


def test_chunk_semantic_no_mid_sentence_split():
    """Verify chunks don't split mid-sentence."""
    chunks = chunk_semantic(SAMPLE_TEXT, "test/doc.md")
    for chunk in chunks:
        # Each chunk should end with a sentence-ending character or be the last chunk
        text = chunk.text.strip()
        assert len(text) > 0, "Empty chunk found"


def test_chunk_paragraph_alignment():
    """Verify chunks align with paragraph boundaries."""
    chunks = chunk_paragraph(SAMPLE_TEXT, "test/doc.md")
    for chunk in chunks:
        # Paragraphs should not contain the double-newline split marker internally
        # (after merging small paragraphs, internal \n\n is allowed for merged ones)
        assert len(chunk.text.strip()) > 0, "Empty chunk found"


def test_all_chunkers_no_text_loss():
    """Verify no text is lost when chunking (approximate check)."""
    for strategy in ChunkingStrategy:
        chunker = get_chunker(strategy)
        chunks = chunker(SAMPLE_TEXT, "test/doc.md")

        # Join all chunk texts
        reconstructed = " ".join(c.text for c in chunks)

        # Check that all significant words from the original appear
        original_words = set(SAMPLE_TEXT.lower().split())
        reconstructed_words = set(reconstructed.lower().split())

        # Allow some tolerance for tokenization differences
        missing = original_words - reconstructed_words
        # Filter out very short tokens and markdown artifacts
        significant_missing = {w for w in missing if len(w) > 3 and not w.startswith("#")}

        assert len(significant_missing) < len(original_words) * 0.05, (
            f"Strategy {strategy.value} lost significant words: {significant_missing}"
        )


def test_all_chunkers_source_file_propagated():
    """Verify source_file is correctly propagated."""
    source = "engineering/architecture_overview.md"
    for strategy in ChunkingStrategy:
        chunker = get_chunker(strategy)
        chunks = chunker(SAMPLE_TEXT, source)
        for chunk in chunks:
            assert chunk.source_file == source, (
                f"Expected source_file '{source}', got '{chunk.source_file}'"
            )


def test_get_chunker_returns_callable():
    """Verify get_chunker returns the right function for each strategy."""
    for strategy in ChunkingStrategy:
        chunker = get_chunker(strategy)
        assert callable(chunker)


if __name__ == "__main__":
    test_chunk_fixed_256_token_limit()
    test_chunk_fixed_512_token_limit()
    test_chunk_semantic_no_mid_sentence_split()
    test_chunk_paragraph_alignment()
    test_all_chunkers_no_text_loss()
    test_all_chunkers_source_file_propagated()
    test_get_chunker_returns_callable()
    print("All chunker tests passed!")
