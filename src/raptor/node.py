"""RaptorNode dataclass: a single node in the RAPTOR tree (leaf or summary)."""

import hashlib
from dataclasses import dataclass, field

from ..chunkers import Chunk


def _stable_chunk_index(node_id: str) -> int:
    """Derive a `Chunk.chunk_index` deterministically from `node_id`.

    Must be stable across process restarts (not just within one run), so
    Python's built-in `hash()` — randomized per-process for strings unless
    seeded before interpreter startup — is not viable. `hashlib.md5` gives
    the same digest for the same input on every run.
    """
    return int(hashlib.md5(node_id.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class RaptorNode:
    """A node in the RAPTOR tree.

    Leaves (level=0) hold the original chunk text.
    Internal nodes hold an LLM-generated summary of their children.
    """

    node_id: str
    text: str
    embedding: list[float]
    level: int
    token_count: int
    source_files: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    def to_chunk(self) -> Chunk:
        """Adapt to the existing `Chunk` shape so downstream scorers work.

        `source_file` joins all provenance files with `;` purely as a
        human-readable display label — it is never parsed. Scorers instead
        use the structured `source_files` list (see `RetrievalScorer`).
        """
        joined = ";".join(self.source_files) if self.source_files else self.node_id
        return Chunk(
            text=self.text,
            source_file=joined,
            chunk_index=_stable_chunk_index(self.node_id),
            chunking_strategy=f"raptor_L{self.level}",
            source_files=self.source_files,
        )

    @property
    def is_leaf(self) -> bool:
        return self.level == 0
