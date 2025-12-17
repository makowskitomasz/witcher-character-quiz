"""CLI for training and exporting character embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from domain.graph_builder import GraphBuilder
from embeddings.hybrid_embeddings import HybridEmbeddings
from embeddings.node2vec_trainer import Node2VecTrainer
from embeddings.quality_analysis import EmbeddingQualityAnalyzer


def main() -> None:
    """Train embeddings and export them."""
    project_root = Path(__file__).parent.parent
    jsonld_path = project_root / "data" / "raw" / "witcher.jsonld"
    output_dir = project_root / "data" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Building graph...")
    builder = GraphBuilder(jsonld_path, trait_threshold=0.7)
    graph = builder.build()
    print(
        f"  Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    print("\nStep 2: Training Node2Vec embeddings...")
    trainer = Node2VecTrainer(graph, dimensions=32, walk_length=10, num_walks=80)
    graph_embeddings = trainer.train()
    print(f"  Trained embeddings for {len(graph_embeddings)} nodes")

    print("\nStep 3: Building hybrid embeddings...")
    hybrid = HybridEmbeddings(graph, graph_embeddings, use_text=False)
    character_embeddings = hybrid.build_character_embeddings()
    print(f"  Created hybrid embeddings for {len(character_embeddings)} characters")

    print("\nStep 4: Quality analysis...")
    analyzer = EmbeddingQualityAnalyzer(character_embeddings, graph=graph)
    reduced = analyzer.reduce_dimensions()
    print(f"  Reduced embeddings to {reduced.shape[1]}D using UMAP")

    clustering = analyzer.analyze_clustering()
    print(f"  Clustering analysis:")
    print(f"    Number of clusters: {clustering['n_clusters']}")
    print(f"    Inertia: {clustering['inertia']:.2f}")

    distances = analyzer.compute_pairwise_distances()
    print(f"\n  Pairwise distances (sample):")
    char_ids = list(character_embeddings.keys())
    for i, char1 in enumerate(char_ids[:3]):
        for char2 in char_ids[i + 1 : min(i + 3, len(char_ids))]:
            dist = distances[char1][char2]
            name1 = graph.nodes[char1].get("name", char1)
            name2 = graph.nodes[char2].get("name", char2)
            print(f"    {name1} <-> {name2}: {dist:.3f}")

    print("\nStep 5: Visualizing embeddings...")
    viz_path = output_dir / "embeddings_umap.png"
    analyzer.visualize(output_path=viz_path, color_by_faction=True, show_labels=True)

    print("\nStep 6: Exporting embeddings...")
    # Export as JSON (convert numpy arrays to lists)
    embeddings_export = {}
    for char_id, emb in character_embeddings.items():
        char_name = graph.nodes[char_id].get("name", char_id)
        embeddings_export[char_id] = {
            "name": char_name,
            "embedding": emb.tolist(),
            "dimension": len(emb),
        }

    embeddings_path = output_dir / "character_embeddings.json"
    with embeddings_path.open("w") as f:
        json.dump(embeddings_export, f, indent=2)
    print(f"  Exported to {embeddings_path}")

    # Export graph embeddings separately
    graph_emb_export = {}
    for node_id, emb in graph_embeddings.items():
        graph_emb_export[node_id] = emb.tolist()

    graph_emb_path = output_dir / "graph_embeddings.json"
    with graph_emb_path.open("w") as f:
        json.dump(graph_emb_export, f, indent=2)
    print(f"  Exported graph embeddings to {graph_emb_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
