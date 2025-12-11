"""Placeholder GraphSAGE trainer (for future use with larger graphs)."""

from __future__ import annotations


class EmbeddingTrainer:
    """Base class for embeddings trainers."""

    def __init__(self) -> None:
        pass

    def train(self) -> None:
        """Run the training procedure."""
        pass


class GraphSAGETrainer(EmbeddingTrainer):
    """Stub for GraphSAGE specific training logic.
    
    GraphSAGE requires PyTorch Geometric or DGL and is overkill for
    the current small graph. This is kept as a placeholder for future
    expansion when the character roster grows.
    """

    def train(self) -> None:  # type: ignore[override]
        """Run the training procedure."""
        raise NotImplementedError(
            "GraphSAGE not implemented. Consider using Node2VecTrainer "
            "for the current graph size, or implement GraphSAGE when "
            "expanding to 50+ characters."
        )
