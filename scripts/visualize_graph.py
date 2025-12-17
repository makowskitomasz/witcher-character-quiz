"""CLI for visualizing the Witcher knowledge graph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from domain.graph_builder import GraphBuilder
from embeddings.graph_visualizer import GraphVisualizer


def main() -> None:
    """Build graph and create visualization."""
    project_root = Path(__file__).parent.parent
    jsonld_path = project_root / "data" / "raw" / "witcher.jsonld"
    output_path = project_root / "data" / "graph_visualization.png"

    print("Building graph from JSON-LD data...")
    builder = GraphBuilder(jsonld_path, trait_threshold=0.7)
    graph = builder.build()

    print(f"Graph statistics:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")

    print("\nCreating multi-panel visualization...")
    visualizer = GraphVisualizer(graph)
    visualizer.visualize_multi_panel(output_path=output_path)

    print(f"\nVisualization saved to {output_path}")


if __name__ == "__main__":
    main()
