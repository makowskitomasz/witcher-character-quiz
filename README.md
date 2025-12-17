# Witcher Character Quiz

A structured groundwork for building an intelligent Witcher character quiz that connects questionnaire responses with knowledge-graph powered character embeddings. This repository sets up the tooling and domain modeling needed for future work on questionnaire design, embedding training, user embedding, and character matching.

## Milestone 1 – Domain Modeling Summary
- Curated the initial list of central Witcher characters to represent in the quiz knowledge base.
- Outlined key personality and behavioral traits that drive quiz logic and future recommendation quality.
- Identified major factions and allegiances to organize character relationships.
- Planned relation types and data flow for mapping characters, factions, and traits.
- Defined an initial graph representation strategy for downstream analysis.
- Reviewed embedding techniques (Node2Vec, GraphSAGE, text-embedding-3-small) to guide later experiments.

## Milestone 2 – Questionnaire Design Summary
- Authored a questionnaire of 12 story-driven, multiple-choice questions spanning decision making, loyalty, moral alignment, interpersonal behavior, and risk tolerance.
- Captured trait weights for each answer, aligning directly with Milestone 1 trait dimensions.
- Formalized the data contract via `questionnaire/schema.py` (Pydantic) and ensured JSON validation through `QuestionnaireLoader`.
- Built the `QuestionnaireTransformer` to map responses to normalized user feature vectors, unlocking downstream embedding pipelines.
- Added a CLI runner (`main.py`) and Makefile commands (`make run`, `make windows_run`) to exercise the pipeline end-to-end.

## Milestone 3 – Embedding Algorithm Selection and Training Summary
- **Graph Construction**: Implemented `GraphBuilder` to construct NetworkX graph from JSON-LD data with nodes for Characters, Factions, and TraitDimensions, and edges for faction membership, character relations, and high-trait connections (threshold 0.7).
- **Node2Vec Training**: Implemented `Node2VecTrainer` using gensim Word2Vec to generate graph embeddings via random walks (32D embeddings, 10-step walks, 80 walks per node).
- **Hybrid Embeddings**: Created `HybridEmbeddings` system that concatenates trait vectors (14D) with graph embeddings (32D) to produce unified character representations (46D total).
- **Character Matching**: Implemented cosine similarity matching between user trait vectors and character hybrid embeddings in `main.py`.
- **Quality Analysis**: Built `EmbeddingQualityAnalyzer` using UMAP for dimensionality reduction visualization and K-means clustering analysis to verify separation and clustering quality.
- **Export Pipeline**: Created `scripts/export_embeddings.py` to train and export embeddings, generate visualizations, and save results to JSON.
- **Dependencies**: Added `umap-learn` for visualization and analysis.

## Milestone 4 – User Embedding Construction Summary
- Introduced `UserEmbeddingBuilder`, which aggregates graph trait embeddings using questionnaire-derived trait weights and outputs L2-normalized hybrid user vectors aligned with character embeddings.
- Updated `main.py` to build user embeddings directly in the questionnaire flow so character matching now happens in the shared latent space (trait + graph portions).
- Added `scripts/test_user_embeddings.py` showcasing multiple synthetic answer profiles, verifying dimensionality, normalization, and producing cosine-based top character matches for each profile.

## Project Structure
- `domain/`: Domain models and graph construction (`graph_builder.py`, `characters.py`, `factions.py`, `traits.py`, `relations.py`)
- `embeddings/`: Embedding training and analysis (`node2vec_trainer.py`, `hybrid_embeddings.py`, `quality_analysis.py`, `graph_visualizer.py`)
- `questionnaire/`: Questionnaire schema and transformation (`schema.py`, `transformer.py`, `loader.py`)
- `scripts/`: CLI tools (`build_graph.py`, `export_embeddings.py`, `visualize_graph.py`)
- `utils/`: Utilities (`io.py` for JSON-LD loading, `config.py`)
- `data/`: Data files (`raw/witcher.jsonld`, `embeddings/` for exported results, `graph.json`)

## Usage

### Running the Quiz
```bash
make run
# or
uv run python main.py
```

### Building the Graph
```bash
make build_graph
# or
uv run python -m scripts.build_graph
```

### Training and Exporting Embeddings
```bash
make export_embeddings
# or
uv run python -m scripts.export_embeddings
```

This will:
1. Build the knowledge graph from JSON-LD data
2. Train Node2Vec embeddings
3. Create hybrid embeddings (trait + graph)
4. Perform quality analysis (UMAP visualization, clustering)
5. Export embeddings to `data/embeddings/`

### Visualizing the Graph
```bash
make visualize_graph
# or
uv run python -m scripts.visualize_graph
```

This creates a multi-panel visualization (`data/graph_visualization.png`) with:
- **Full Graph**: Complete knowledge graph with all nodes (Characters, Factions, Traits) color-coded by type
- **Character Subgraph**: Focused view showing only characters, their relations, and faction memberships
- **Trait Heatmap**: Character × Trait matrix with hierarchical clustering to show trait similarity patterns

## How to Extend

### Extending the Questionnaire
- Update `questionnaire/questionnaire.json` by appending new questions that follow the existing schema (unique IDs, text, options with `trait_mapping` weights). Keep trait names aligned with the canonical list in `questionnaire/transformer.py`.
- Run `QuestionnaireLoader.from_default_path().load()` to validate the JSON through the Pydantic models in `questionnaire/schema.py`. Validation errors point directly to malformed entries.
- If you introduce new trait dimensions, expand `TRAIT_ORDER` inside `questionnaire/transformer.py` to keep feature vectors aligned and reproducible across experiments.

### Adding Characters
- Add character entries to `data/raw/witcher.jsonld` following the existing schema with `traitValues` and `relation` fields.
- Run `make build_graph` to rebuild the graph.
- Run `make export_embeddings` to retrain embeddings with the new character.

### Adjusting Embedding Parameters
- Modify `Node2VecTrainer` parameters in `scripts/export_embeddings.py` (dimensions, walk_length, num_walks).
- Adjust trait threshold in `GraphBuilder` (default 0.7) to control `hasHighTrait` edge creation.
- Tune UMAP parameters in `EmbeddingQualityAnalyzer` for different visualization results.
