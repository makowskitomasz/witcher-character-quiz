"""CLI for building the Witcher domain graph."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from domain.graph_builder import GraphBuilder


def main() -> None:
    """Build and inspect the domain graph."""
    project_root = Path(__file__).parent.parent
    jsonld_path = project_root / "data" / "raw" / "witcher.jsonld"

    print("Building graph from JSON-LD data...")
    builder = GraphBuilder(jsonld_path, trait_threshold=0.7)
    graph = builder.build()

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")

    print(f"\nNode Types:")
    node_types: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")

    print(f"\nEdge Types:")
    edge_types: dict[str, int] = {}
    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get("relation_type", "Unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")

    print(f"\nCharacters:")
    for char_id in builder.get_character_nodes():
        node_data = graph.nodes[char_id]
        char_name = node_data.get("name", char_id)
        print(f"  {char_name} ({char_id})")

    # Save graph to JSON for inspection
    output_path = project_root / "data" / "graph.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph_data = nx.node_link_data(graph)
    with output_path.open("w") as f:
        json.dump(graph_data, f, indent=2, default=str)

    print(f"\nGraph saved to {output_path}")


if __name__ == "__main__":
    main()
