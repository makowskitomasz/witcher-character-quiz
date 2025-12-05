"""Embedding training interfaces and implementations."""

from __future__ import annotations


class EmbeddingTrainer:
    """Base class for embeddings trainers."""

    def __init__(self) -> None:
        pass

    def train(self) -> None:
        """Run the training procedure."""

        pass


__all__ = ["EmbeddingTrainer"]
