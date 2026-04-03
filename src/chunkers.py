"""Chunking strategies for splitting documents into retrieval units."""

import os
from dataclasses import dataclass

import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import ChunkingStrategy, EMBEDDING_MODEL


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int
    chunking_strategy: str


_tokenizer = tiktoken.get_encoding("cl100k_base")

# Lazy-loaded embedding model for semantic chunking
_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer(EMBEDDING_MODEL)
    return _semantic_model


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def chunk_fixed_256(text: str, source_file: str) -> list[Chunk]:
    """Split text into 256-token windows with 50-token overlap."""
    return _chunk_fixed(text, source_file, window=256, overlap=50, strategy="fixed_256")


def chunk_fixed_512(text: str, source_file: str) -> list[Chunk]:
    """Split text into 512-token windows with 100-token overlap."""
    return _chunk_fixed(text, source_file, window=512, overlap=100, strategy="fixed_512")


def _chunk_fixed(text: str, source_file: str, window: int, overlap: int, strategy: str) -> list[Chunk]:
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(tokens):
        end = min(start + window, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(Chunk(
            text=chunk_text,
            source_file=source_file,
            chunk_index=chunk_index,
            chunking_strategy=strategy,
        ))
        chunk_index += 1
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_semantic(text: str, source_file: str) -> list[Chunk]:
    """Split text by semantic similarity between consecutive sentences."""
    model = _get_semantic_model()
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return [Chunk(
            text=text,
            source_file=source_file,
            chunk_index=0,
            chunking_strategy="semantic",
        )]

    embeddings = model.encode(sentences)

    # Compute cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        a = embeddings[i]
        b = embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        similarities.append(float(cos_sim))

    # Group sentences into chunks based on similarity threshold
    threshold = 0.5
    groups = []
    current_group = [0]

    for i, sim in enumerate(similarities):
        if sim < threshold:
            groups.append(current_group)
            current_group = [i + 1]
        else:
            current_group.append(i + 1)
    groups.append(current_group)

    # Enforce minimum chunk size of 2 sentences
    merged_groups = []
    for idx, group in enumerate(groups):
        if len(group) < 2 and merged_groups:
            # Merge with shorter adjacent group
            has_next = idx + 1 < len(groups)
            merge_with_prev = not has_next or len(merged_groups[-1]) <= len(groups[idx + 1])
            if merge_with_prev:
                merged_groups[-1].extend(group)
            else:
                merged_groups.append(group)
        else:
            merged_groups.append(group)

    # Handle case where last group has only 1 sentence
    if len(merged_groups) > 1 and len(merged_groups[-1]) < 2:
        merged_groups[-2].extend(merged_groups[-1])
        merged_groups.pop()

    chunks = []
    for chunk_index, group in enumerate(merged_groups):
        chunk_text = " ".join(sentences[i] for i in group)
        chunks.append(Chunk(
            text=chunk_text,
            source_file=source_file,
            chunk_index=chunk_index,
            chunking_strategy="semantic",
        ))
    return chunks


def chunk_paragraph(text: str, source_file: str) -> list[Chunk]:
    """Split on double newlines, merging short paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge short paragraphs (< 50 tokens) with the next paragraph
    merged = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            buffer = buffer + "\n\n" + para
        else:
            buffer = para

        if _count_tokens(buffer) >= 50:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer)

    chunks = []
    for chunk_index, para_text in enumerate(merged):
        chunks.append(Chunk(
            text=para_text,
            source_file=source_file,
            chunk_index=chunk_index,
            chunking_strategy="paragraph",
        ))
    return chunks


_CHUNKER_MAP = {
    ChunkingStrategy.FIXED_256: chunk_fixed_256,
    ChunkingStrategy.FIXED_512: chunk_fixed_512,
    ChunkingStrategy.SEMANTIC: chunk_semantic,
    ChunkingStrategy.PARAGRAPH: chunk_paragraph,
}


def get_chunker(strategy: ChunkingStrategy):
    """Return the appropriate chunking function for the given strategy."""
    return _CHUNKER_MAP[strategy]


def chunk_corpus(strategy: ChunkingStrategy, corpus_dir: str) -> list[Chunk]:
    """Chunk all .md files in the corpus directory using the given strategy."""
    chunker = get_chunker(strategy)
    all_chunks = []

    for root, _dirs, files in os.walk(corpus_dir):
        for filename in sorted(files):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, corpus_dir).replace("\\", "/")
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunker(text, rel_path)
            all_chunks.extend(chunks)

    return all_chunks
