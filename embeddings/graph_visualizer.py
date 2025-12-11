"""Graph visualization tools for the Witcher knowledge graph."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from questionnaire.transformer import TRAIT_ORDER


class GraphVisualizer:
    """Visualize the Witcher knowledge graph with multiple views."""

    def __init__(self, graph: nx.Graph) -> None:
        """Initialize visualizer with a NetworkX graph.
        
        Args:
            graph: NetworkX graph from GraphBuilder
        """
        self.graph = graph

    def visualize_multi_panel(
        self,
        output_path: Path | str | None = None,
        figsize: tuple[int, int] = (18, 12),
    ) -> None:
        """Create multi-panel visualization: full graph, character subgraph, trait heatmap.
        
        Args:
            output_path: Path to save figure (if None, displays interactively)
            figsize: Figure size (width, height)
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Full graph
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_full_graph(ax1)
        
        # Panel 2: Character-only subgraph
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_character_subgraph(ax2)
        
        # Panel 3: Trait heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_trait_heatmap(ax3)
        
        plt.suptitle("Witcher Knowledge Graph Visualization", fontsize=16, fontweight="bold", y=0.98)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

    def _plot_full_graph(self, ax) -> None:
        """Plot full graph with all node types."""
        # Separate nodes by type
        characters = [
            n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "Character"
        ]
        factions = [
            n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "Faction"
        ]
        traits = [
            n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "TraitDimension"
        ]
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Draw edges
        # Separate edge types for different styling
        char_char_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relation_type") in ["friend_of", "lover_of", "destiny_of", "ward_of", "adoptive_child_of"]
        ]
        faction_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relation_type") == "hasFaction"
        ]
        trait_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relation_type") == "hasHighTrait"
        ]
        
        # Draw edges (traits first so they're in background)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=trait_edges,
            alpha=0.2,
            width=0.5,
            edge_color="lightgray",
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=faction_edges,
            alpha=0.6,
            width=1.5,
            edge_color="green",
            style="dashed",
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=char_char_edges,
            alpha=0.8,
            width=2.0,
            edge_color="red",
            ax=ax,
        )
        
        # Draw nodes by type
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=characters,
            node_color="steelblue",
            node_size=800,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=factions,
            node_color="forestgreen",
            node_size=600,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=traits,
            node_color="orange",
            node_size=300,
            alpha=0.7,
            ax=ax,
        )
        
        # Draw labels for characters and factions only
        char_faction_labels = {n: self.graph.nodes[n].get("name", n) for n in characters + factions}
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=char_faction_labels,
            font_size=8,
            font_weight="bold",
            ax=ax,
        )
        
        ax.set_title("Full Knowledge Graph", fontsize=12, fontweight="bold")
        ax.axis("off")
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=10, label="Characters"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="forestgreen", markersize=10, label="Factions"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Traits"),
            plt.Line2D([0], [0], color="red", linewidth=2, label="Character Relations"),
            plt.Line2D([0], [0], color="green", linewidth=2, linestyle="--", label="Faction Membership"),
            plt.Line2D([0], [0], color="lightgray", linewidth=1, label="High Traits"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    def _plot_character_subgraph(self, ax) -> None:
        """Plot character-only subgraph with their relations."""
        # Get character nodes
        characters = [
            n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "Character"
        ]
        
        # Create subgraph with characters and their immediate neighbors (factions)
        char_subgraph_nodes = set(characters)
        for char in characters:
            for neighbor in self.graph.neighbors(char):
                neighbor_data = self.graph.nodes.get(neighbor, {})
                if neighbor_data.get("type") == "Faction":
                    char_subgraph_nodes.add(neighbor)
        
        subgraph = self.graph.subgraph(char_subgraph_nodes)
        
        # Layout
        pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
        
        # Separate edges
        char_char_edges = []
        faction_edges = []
        
        for u, v, d in subgraph.edges(data=True):
            rel_type = d.get("relation_type", "")
            if rel_type in ["friend_of", "lover_of", "destiny_of", "ward_of", "adoptive_child_of"]:
                char_char_edges.append((u, v))
            elif rel_type == "hasFaction":
                faction_edges.append((u, v))
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=faction_edges,
            alpha=0.6,
            width=2.0,
            edge_color="green",
            style="dashed",
            ax=ax,
        )
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=char_char_edges,
            alpha=0.8,
            width=2.5,
            edge_color="red",
            ax=ax,
        )
        
        # Draw nodes
        char_nodes = [n for n in subgraph.nodes() if n in characters]
        faction_nodes = [n for n in subgraph.nodes() if n not in characters]
        
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=char_nodes,
            node_color="steelblue",
            node_size=1000,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=faction_nodes,
            node_color="forestgreen",
            node_size=700,
            alpha=0.9,
            ax=ax,
        )
        
        # Draw labels
        labels = {n: self.graph.nodes[n].get("name", n) for n in subgraph.nodes()}
        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels=labels,
            font_size=9,
            font_weight="bold",
            ax=ax,
        )
        
        # Add edge labels for character relations
        edge_labels = {}
        for u, v, d in subgraph.edges(data=True):
            rel_type = d.get("relation_type", "")
            if rel_type in ["friend_of", "lover_of", "destiny_of", "ward_of", "adoptive_child_of"]:
                # Shorten labels
                rel_type_short = rel_type.replace("_of", "").replace("_", " ").title()
                edge_labels[(u, v)] = rel_type_short
        
        nx.draw_networkx_edge_labels(
            subgraph,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            ax=ax,
        )
        
        ax.set_title("Character Relations & Factions", fontsize=12, fontweight="bold")
        ax.axis("off")

    def _plot_trait_heatmap(self, ax) -> None:
        """Plot trait heatmap: Characters × Traits matrix."""
        # Get all characters
        characters = [
            n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "Character"
        ]
        
        # Build trait matrix
        trait_matrix = []
        char_names = []
        
        for char_id in characters:
            node_data = self.graph.nodes[char_id]
            char_name = node_data.get("name", char_id)
            char_names.append(char_name)
            
            trait_vector = node_data.get("trait_vector", np.zeros(len(TRAIT_ORDER)))
            trait_matrix.append(trait_vector)
        
        trait_matrix = np.array(trait_matrix)
        
        # Hierarchical clustering on rows (characters)
        if len(characters) > 1:
            char_distances = pdist(trait_matrix, metric="euclidean")
            char_linkage = linkage(char_distances, method="ward")
            char_dendro = dendrogram(char_linkage, no_plot=True)
            char_order = char_dendro["leaves"]
        else:
            char_order = [0]
        
        # Hierarchical clustering on columns (traits)
        if len(TRAIT_ORDER) > 1:
            trait_distances = pdist(trait_matrix.T, metric="euclidean")
            trait_linkage = linkage(trait_distances, method="ward")
            trait_dendro = dendrogram(trait_linkage, no_plot=True)
            trait_order = trait_dendro["leaves"]
        else:
            trait_order = list(range(len(TRAIT_ORDER)))
        
        # Reorder matrix
        reordered_matrix = trait_matrix[char_order, :][:, trait_order]
        reordered_char_names = [char_names[i] for i in char_order]
        reordered_trait_names = [TRAIT_ORDER[i] for i in trait_order]
        
        # Plot heatmap
        im = ax.imshow(reordered_matrix, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(len(reordered_trait_names)))
        ax.set_yticks(range(len(reordered_char_names)))
        ax.set_xticklabels(reordered_trait_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(reordered_char_names, fontsize=9, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Trait Value", fontsize=9)
        
        # Add grid
        ax.set_xticks(np.arange(len(reordered_trait_names)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(reordered_char_names)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        
        ax.set_title("Character Trait Heatmap (Clustered)", fontsize=12, fontweight="bold")

