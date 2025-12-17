"""Node2Vec trainer for graph embeddings."""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
from gensim.models import Word2Vec


class Node2VecTrainer:
    """Train Node2Vec embeddings using gensim Word2Vec."""

    def __init__(
        self,
        graph: nx.Graph,
        dimensions: int = 32,
        walk_length: int = 10,
        num_walks: int = 80,
        window: int = 5,
        workers: int = 1,
        seed: int = 42,
    ) -> None:
        """Initialize Node2Vec trainer.

        Args:
            graph: NetworkX graph to embed
            dimensions: Embedding dimension size
            walk_length: Length of random walks
            num_walks: Number of walks per node
            window: Context window size for Word2Vec
            workers: Number of worker threads
            seed: Random seed for reproducibility
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.workers = workers
        self.seed = seed
        self.model: Word2Vec | None = None
        self.embeddings: Dict[str, np.ndarray] | None = None

    def _generate_random_walks(self) -> List[List[str]]:
        """Generate random walks on the graph.

        Returns:
            List of walks, where each walk is a list of node IDs
        """
        walks = []
        nodes = list(self.graph.nodes())

        for _ in range(self.num_walks):
            np.random.seed(self.seed + len(walks))
            # Shuffle nodes for each iteration
            np.random.shuffle(nodes)
            for start_node in nodes:
                walk = [start_node]
                current = start_node

                for _ in range(self.walk_length - 1):
                    neighbors = list(self.graph.neighbors(current))
                    if not neighbors:
                        break
                    # Uniform random walk (Node2Vec uses biased walks, but for simplicity we use uniform)
                    current = np.random.choice(neighbors)
                    walk.append(current)

                walks.append(walk)

        return walks

    def train(self) -> Dict[str, np.ndarray]:
        """Train Node2Vec embeddings.

        Returns:
            Dictionary mapping node ID to embedding vector
        """
        # Generate random walks
        walks = self._generate_random_walks()

        # Convert walks to strings (gensim expects list of lists of strings)
        walks_str = [[str(node) for node in walk] for walk in walks]

        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=walks_str,
            vector_size=self.dimensions,
            window=self.window,
            min_count=1,  # Include all nodes even if they appear once
            workers=self.workers,
            seed=self.seed,
            sg=1,  # Skip-gram model
        )

        # Extract embeddings
        self.embeddings = {}
        for node in self.graph.nodes():
            node_str = str(node)
            if node_str in self.model.wv:
                self.embeddings[node] = self.model.wv[node_str]
            else:
                # Fallback: zero vector if node not in vocabulary
                self.embeddings[node] = np.zeros(self.dimensions, dtype=np.float32)

        return self.embeddings

    def get_embedding(self, node_id: str) -> np.ndarray:
        """Get embedding for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            Embedding vector
        """
        if self.embeddings is None:
            raise ValueError("Model not trained yet. Call train() first.")
        if node_id not in self.embeddings:
            raise ValueError(f"Node {node_id} not found in embeddings.")
        return self.embeddings[node_id]
