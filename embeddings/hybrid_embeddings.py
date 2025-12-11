"""Hybrid embedding system combining trait vectors and graph embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
import networkx as nx

from questionnaire.transformer import TRAIT_ORDER


class HybridEmbeddings:
    """Combines trait vectors and graph embeddings into unified character representations."""

    def __init__(
        self,
        graph: nx.Graph,
        graph_embeddings: Dict[str, np.ndarray],
        use_text: bool = False,
        text_encoder=None,
    ) -> None:
        """Initialize hybrid embedding system.
        
        Args:
            graph: NetworkX graph with character nodes
            graph_embeddings: Dictionary mapping node ID to graph embedding vector
            use_text: Whether to include text embeddings (requires text_encoder)
            text_encoder: TextEncoder instance (optional, only if use_text=True)
        """
        self.graph = graph
        self.graph_embeddings = graph_embeddings
        self.use_text = use_text
        self.text_encoder = text_encoder
        self.character_embeddings: Dict[str, np.ndarray] | None = None

    def build_character_embeddings(self) -> Dict[str, np.ndarray]:
        """Build hybrid embeddings for all characters.
        
        Returns:
            Dictionary mapping character ID to hybrid embedding vector
        """
        character_nodes = [
            n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "Character"
        ]
        
        self.character_embeddings = {}
        
        for char_id in character_nodes:
            # Get trait vector (14D)
            node_data = self.graph.nodes[char_id]
            trait_vector = node_data.get("trait_vector", np.zeros(len(TRAIT_ORDER)))
            
            # Get graph embedding (32D or whatever dimension was used)
            graph_emb = self.graph_embeddings.get(char_id, np.zeros(32))
            
            # Optionally get text embedding
            if self.use_text and self.text_encoder:
                char_name = node_data.get("name", "")
                text_emb = self.text_encoder.encode(char_name)
            else:
                text_emb = np.array([])
            
            # Concatenate: [trait_vector | graph_embedding | text_embedding]
            hybrid = np.concatenate([trait_vector, graph_emb, text_emb])
            self.character_embeddings[char_id] = hybrid
        
        return self.character_embeddings

    def get_character_embedding(self, character_id: str) -> np.ndarray:
        """Get hybrid embedding for a specific character.
        
        Args:
            character_id: Character node ID
            
        Returns:
            Hybrid embedding vector
        """
        if self.character_embeddings is None:
            self.build_character_embeddings()
        
        if character_id not in self.character_embeddings:
            raise ValueError(f"Character {character_id} not found in embeddings.")
        
        return self.character_embeddings[character_id]

    def match_user_to_character(
        self, user_trait_vector: np.ndarray, method: str = "cosine"
    ) -> list[tuple[str, float]]:
        """Match user trait vector to characters using similarity.
        
        Args:
            user_trait_vector: User's trait vector (14D)
            method: Similarity method ("cosine" or "euclidean")
            
        Returns:
            List of (character_id, similarity_score) tuples, sorted by score descending
        """
        if self.character_embeddings is None:
            self.build_character_embeddings()
        
        # For user matching, we only use the trait vector portion
        # Pad user vector to match hybrid embedding dimension
        trait_dim = len(TRAIT_ORDER)
        graph_dim = len(next(iter(self.graph_embeddings.values())))
        text_dim = 1536 if self.use_text else 0
        
        # Create user embedding: [user_trait_vector | zeros for graph | zeros for text]
        user_hybrid = np.concatenate([
            user_trait_vector,
            np.zeros(graph_dim),
            np.zeros(text_dim),
        ])
        
        scores = []
        for char_id, char_emb in self.character_embeddings.items():
            if method == "cosine":
                # Cosine similarity
                dot_product = np.dot(user_hybrid, char_emb)
                norm_user = np.linalg.norm(user_hybrid)
                norm_char = np.linalg.norm(char_emb)
                if norm_user > 0 and norm_char > 0:
                    similarity = dot_product / (norm_user * norm_char)
                else:
                    similarity = 0.0
            elif method == "euclidean":
                # Euclidean distance (convert to similarity: 1 / (1 + distance))
                distance = np.linalg.norm(user_hybrid - char_emb)
                similarity = 1.0 / (1.0 + distance)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            scores.append((char_id, float(similarity)))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

