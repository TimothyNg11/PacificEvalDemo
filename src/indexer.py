"""Index building for ChromaDB (vector) and BM25 (keyword) search."""

import os

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .chunkers import Chunk, chunk_corpus
from .config import ChunkingStrategy, EMBEDDING_MODEL, CHROMA_DIR


class CorpusIndex:
    """Holds both a ChromaDB collection and a BM25 index for a set of chunks."""

    def __init__(self, chunks: list[Chunk], config_name: str):
        self.chunks = chunks
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Build ChromaDB collection
        persist_dir = os.path.join(CHROMA_DIR, config_name)
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            name=config_name,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            # Embed and add all chunks
            texts = [c.text for c in chunks]
            embeddings = self.embedding_model.encode(texts).tolist()
            ids = [f"{c.source_file}_{c.chunk_index}" for c in chunks]
            metadatas = [
                {"source_file": c.source_file, "chunk_index": c.chunk_index}
                for c in chunks
            ]
            # ChromaDB has batch size limits; add in batches of 5000
            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i : i + batch_size],
                    documents=texts[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                )

        # Build BM25 index
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)


class IndexBuilder:
    """Builds indexes for all chunking strategies."""

    def build_all_indexes(self, corpus_dir: str) -> dict[str, CorpusIndex]:
        indexes = {}
        for strategy in ChunkingStrategy:
            config_name = strategy.value
            persist_dir = os.path.join(CHROMA_DIR, config_name)

            print(f"Building index for chunking strategy: {config_name}...")
            chunks = chunk_corpus(strategy, corpus_dir)
            print(f"  Chunked corpus into {len(chunks)} chunks")

            index = CorpusIndex(chunks, config_name)
            indexes[config_name] = index
            print(f"  Index built: {index.collection.count()} vectors in ChromaDB")

        return indexes
