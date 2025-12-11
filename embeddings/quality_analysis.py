"""Quality analysis tools for embeddings using dimensionality reduction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP


class EmbeddingQualityAnalyzer:
    """Analyze embedding quality using dimensionality reduction and clustering."""

    def __init__(
        self,
        character_embeddings: Dict[str, np.ndarray],
        graph: nx.Graph | None = None,
        n_components: int = 2,
        random_state: int = 42,
    ) -> None:
        """Initialize quality analyzer.
        
        Args:
            character_embeddings: Dictionary mapping character ID to embedding
            graph: Optional NetworkX graph for extracting metadata (factions, etc.)
            n_components: Number of dimensions for UMAP reduction (2 for visualization)
            random_state: Random seed
        """
        self.character_embeddings = character_embeddings
        self.graph = graph
        self.n_components = n_components
        self.random_state = random_state
        self.umap_model: UMAP | None = None
        self.reduced_embeddings: np.ndarray | None = None

    def reduce_dimensions(self) -> np.ndarray:
        """Reduce embeddings to 2D using UMAP.
        
        Returns:
            Array of shape (n_characters, n_components) with reduced embeddings
        """
        # Prepare data matrix
        char_ids = list(self.character_embeddings.keys())
        embeddings_matrix = np.array([self.character_embeddings[cid] for cid in char_ids])
        
        # Fit UMAP
        self.umap_model = UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=min(5, len(char_ids) - 1),  # Adjust for small datasets
        )
        self.reduced_embeddings = self.umap_model.fit_transform(embeddings_matrix)
        
        return self.reduced_embeddings

    def visualize(
        self,
        output_path: Path | str | None = None,
        color_by_faction: bool = True,
        show_labels: bool = True,
    ) -> None:
        """Visualize reduced embeddings.
        
        Args:
            output_path: Path to save figure (if None, displays interactively)
            color_by_faction: Whether to color points by faction membership
            show_labels: Whether to show character names
        """
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        char_ids = list(self.character_embeddings.keys())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if color_by_faction and self.graph:
            # Color by faction
            faction_colors = {}
            color_map = plt.cm.Set3
            faction_idx = 0
            
            for char_id in char_ids:
                node_data = self.graph.nodes.get(char_id, {})
                factions = []
                
                # Find faction neighbors
                for neighbor in self.graph.neighbors(char_id):
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    if neighbor_data.get("type") == "Faction":
                        factions.append(neighbor)
                
                if factions:
                    # Use first faction for coloring
                    faction_id = factions[0]
                    if faction_id not in faction_colors:
                        faction_colors[faction_id] = color_map(faction_idx / max(1, len(faction_colors)))
                        faction_idx += 1
                    color = faction_colors[faction_id]
                else:
                    color = "gray"
                
                x, y = self.reduced_embeddings[char_ids.index(char_id)]
                ax.scatter(x, y, c=[color], s=100, alpha=0.7)
                
                if show_labels:
                    char_name = node_data.get("name", char_id)
                    ax.annotate(char_name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
        else:
            # Single color
            ax.scatter(
                self.reduced_embeddings[:, 0],
                self.reduced_embeddings[:, 1],
                s=100,
                alpha=0.7,
            )
            
            if show_labels:
                for idx, char_id in enumerate(char_ids):
                    node_data = self.graph.nodes.get(char_id, {}) if self.graph else {}
                    char_name = node_data.get("name", char_id)
                    x, y = self.reduced_embeddings[idx]
                    ax.annotate(char_name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
        
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.set_title("Character Embeddings (UMAP Reduction)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

    def analyze_clustering(self, n_clusters: int | None = None) -> Dict[str, any]:
        """Analyze clustering quality.
        
        Args:
            n_clusters: Number of clusters (if None, uses number of factions)
            
        Returns:
            Dictionary with clustering metrics and assignments
        """
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        char_ids = list(self.character_embeddings.keys())
        
        # Determine number of clusters
        if n_clusters is None:
            if self.graph:
                # Count unique factions
                factions = set()
                for char_id in char_ids:
                    for neighbor in self.graph.neighbors(char_id):
                        neighbor_data = self.graph.nodes.get(neighbor, {})
                        if neighbor_data.get("type") == "Faction":
                            factions.add(neighbor)
                n_clusters = max(2, len(factions))
            else:
                n_clusters = 3  # Default
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(self.reduced_embeddings)
        
        # Calculate within-cluster sum of squares
        inertia = kmeans.inertia_
        
        # Group characters by cluster
        clusters: Dict[int, List[str]] = {}
        for idx, char_id in enumerate(char_ids):
            cluster_id = cluster_labels[idx]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(char_id)
        
        return {
            "n_clusters": n_clusters,
            "inertia": float(inertia),
            "cluster_assignments": {cid: int(cluster_labels[char_ids.index(cid)]) for cid in char_ids},
            "clusters": {str(k): v for k, v in clusters.items()},
        }

    def compute_pairwise_distances(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise cosine distances between all characters.
        
        Returns:
            Dictionary mapping (char1, char2) to distance
        """
        char_ids = list(self.character_embeddings.keys())
        distances: Dict[str, Dict[str, float]] = {}
        
        for char1 in char_ids:
            distances[char1] = {}
            emb1 = self.character_embeddings[char1]
            norm1 = np.linalg.norm(emb1)
            
            for char2 in char_ids:
                if char1 == char2:
                    distances[char1][char2] = 0.0
                else:
                    emb2 = self.character_embeddings[char2]
                    norm2 = np.linalg.norm(emb2)
                    
                    if norm1 > 0 and norm2 > 0:
                        cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
                        cosine_dist = 1.0 - cosine_sim
                    else:
                        cosine_dist = 1.0
                    
                    distances[char1][char2] = float(cosine_dist)
        
        return distances

