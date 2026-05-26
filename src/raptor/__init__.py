"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Reimplementation of Sarthi et al. (ICLR 2024) plus a query-conditional
tree traversal contribution.

Submodules are imported directly to avoid forcing optional dependencies
(`umap-learn`, `scikit-learn`) on consumers that only need lightweight
symbols like `RaptorNode` or the cache helpers.
"""
