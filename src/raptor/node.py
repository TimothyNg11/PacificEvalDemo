"""RaptorNode dataclass: a single node in the RAPTOR tree (leaf or summary)."""

from dataclasses import dataclass, field

from ..chunkers import Chunk


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
    leaf_indices: list[int] = field(default_factory=list)
    cluster_id: int | None = None

    def to_chunk(self) -> Chunk:
        """Adapt to the existing `Chunk` shape so downstream scorers work.

        `source_file` joins all provenance files with `;` so that
        `RetrievalScorer.context_recall` (which does set membership on
        `chunk.source_file`) still matches gold_source_ids that overlap.
        """
        joined = ";".join(self.source_files) if self.source_files else self.node_id
        return Chunk(
            text=self.text,
            source_file=joined,
            chunk_index=hash(self.node_id) & 0x7FFFFFFF,
            chunking_strategy=f"raptor_L{self.level}",
        )

    @property
    def is_leaf(self) -> bool:
        return self.level == 0
