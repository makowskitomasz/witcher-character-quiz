"""User embedding construction utilities."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .transformer import TRAIT_ORDER, QuestionnaireTransformer


class UserEmbeddingBuilder:
    """Constructs normalized user embeddings compatible with character vectors.

    The builder expects trait embeddings that already live in the same latent space
    as character graph embeddings (e.g. Node2Vec). User trait weights obtained from
    the questionnaire transformer are used as linear coefficients for these trait
    embeddings. The resulting hybrid vector concatenates the original trait feature
    vector with the aggregated graph embedding and is L2-normalized so cosine
    similarity can be computed directly against character embeddings.
    """

    def __init__(
        self,
        trait_embeddings: Mapping[str, Sequence[float]],
        trait_order: Sequence[str] | None = None,
        trait_prefix: str = "wc:",
    ) -> None:
        if not trait_embeddings:
            raise ValueError(
                "UserEmbeddingBuilder requires at least one trait embedding."
            )

        self.trait_order = (
            list(trait_order) if trait_order is not None else list(TRAIT_ORDER)
        )
        if not self.trait_order:
            raise ValueError("Trait order must contain at least one trait dimension.")

        self.trait_prefix = trait_prefix
        self._trait_embeddings: dict[str, np.ndarray] = {}
        self.graph_dimension: int | None = None

        for trait in self.trait_order:
            node_id = f"{self.trait_prefix}{trait}"
            raw_vector = trait_embeddings.get(node_id)
            if raw_vector is None:
                continue
            vector = np.asarray(raw_vector, dtype=float)
            if vector.ndim != 1:
                raise ValueError(
                    f"Trait embedding for '{trait}' must be one-dimensional."
                )
            if self.graph_dimension is None:
                self.graph_dimension = vector.shape[0]
            elif vector.shape[0] != self.graph_dimension:
                raise ValueError(
                    f"Inconsistent trait embedding dimensions detected for '{trait}'. "
                    f"Expected {self.graph_dimension}, received {vector.shape[0]}."
                )
            self._trait_embeddings[trait] = vector

        graph_dim = self.graph_dimension
        if graph_dim is None:
            raise ValueError(
                "None of the requested traits were found in the provided embeddings."
            )

        self.graph_dimension = int(graph_dim)
        self.embedding_dimension = self.graph_dimension + len(self.trait_order)

    def build_from_answers(
        self, answers: Mapping[str, str], transformer: QuestionnaireTransformer
    ) -> np.ndarray:
        """Convenience helper that converts answers to a user embedding."""
        feature_vector = transformer.transform(answers)
        return self.build_from_feature_vector(feature_vector)

    def build_from_feature_vector(self, feature_vector: Sequence[float]) -> np.ndarray:
        """Construct a normalized user embedding from a trait feature vector."""
        vector = np.asarray(feature_vector, dtype=float)
        if vector.shape[0] != len(self.trait_order):
            raise ValueError(
                "Feature vector length does not match configured trait order. "
                f"Expected {len(self.trait_order)}, received {vector.shape[0]}."
            )

        graph_component = np.zeros(self.graph_dimension, dtype=float)
        for idx, weight in enumerate(vector):
            if weight == 0:
                continue
            trait = self.trait_order[idx]
            trait_embedding = self._trait_embeddings.get(trait)
            if trait_embedding is None:
                continue
            graph_component += weight * trait_embedding

        combined = np.concatenate([vector, graph_component])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        print(combined)
        return combined
