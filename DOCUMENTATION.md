# Witcher Character Quiz - Documentation

This document provides a comprehensive overview of the embeddings-based psychotest system for matching users with Witcher characters.

## 1. Domain Model

The system is built on a knowledge graph representing the Witcher universe.

### Entities
- **Characters**: Central figures from the Witcher series (Geralt, Ciri, Yennefer, Triss, Dandelion, Vesemir). Each character has a set of 14 trait values (0.0 to 1.0).
- **Factions**: Major groups in the world (Witchers, Mages, Commoners, Nobility).
- **Trait Dimensions**: 14 personality and behavioral traits used for evaluation.

### Trait Dimensions
1.  **Openness**: Curiosity and willingness to explore.
2.  **Conscientiousness**: Discipline and reliability.
3.  **Extraversion**: Sociability and outgoingness.
4.  **Agreeableness**: Cooperation and empathy.
5.  **Emotional Stability**: Resilience under stress.
6.  **Pragmatism**: Practicality vs. idealism.
7.  **Bravery**: Courage under pressure.
8.  **Moral Flexibility**: Tolerance for ethical ambiguity.
9.  **Loyalty**: Commitment to allies and causes.
10. **Empathy**: Compassion for others.
11. **Independence**: Self-reliance and autonomy.
12. **Ambition**: Drive for mastery or power.
13. **Adaptability**: Resilience to change.
14. **Impulsiveness**: Acting on gut feelings vs. deliberation.

## 2. Questionnaire

The questionnaire consists of 12 story-driven questions.

### Logic
- Each option in a question is mapped to specific trait weights.
- **Transformation**: Answers are processed by `QuestionnaireTransformer`, which aggregates weights and produces a normalized 14D trait vector for the user.
- **Weights**: Traits are balanced across questions to ensure comprehensive coverage without over-sensitivity in a single dimension.

## 3. Embedding Workflow

The matching system uses a hybrid embedding approach.

### Graph Construction
- A graph is built using `GraphBuilder` from the `witcher.jsonld` data.
- Edges represent:
    - **Faction Membership**: `(Character) -[hasFaction]-> (Faction)`
    - **Character Relations**: `(Character) -[friend_of/lover_of/etc]-> (Character)`
    - **High Traits**: `(Character) -[hasHighTrait]-> (TraitDimension)` (for traits > 0.7).

### Feature Generation
- **Node2Vec**: Random walks on the graph generate 32D structural embeddings for each node.
- **Hybrid Construction**: For each character, the 14D trait vector is concatenated with the 32D graph embedding, resulting in a **46D hybrid embedding**.

## 4. Matching Algorithm

Matching happens in the 46D latent space.

### Process
1.  **User Trait Vector**: Obtained from the questionnaire (14D).
2.  **User Graph Embedding**: The `UserEmbeddingBuilder` calculates a weighted average of the graph embeddings of trait nodes, based on the user's trait scores.
3.  **User Hybrid Embedding**: Concatenate the 14D user traits with the 32D calculated graph embedding (46D).
4.  **Similarity Scoring**: Cosine similarity is calculated between the 46D user embedding and all 46D character embeddings.
5.  **Explanation**: The `ResultExplainer` compares the user's top traits with the matched character's traits to generate a natural language summary.

## 5. Usage Examples

### Running the Quiz
```bash
make run
```

### Visualizing the Results
Training and exporting embeddings generates a UMAP visualization in `data/embeddings/embeddings_umap.png`.
```bash
make export_embeddings
```

### Extending the System
- **Adding Characters**: Edit `data/raw/witcher.jsonld`.
- **Adding Questions**: Edit `questionnaire/questionnaire.json`.
- **Validation**: Run the consistency check to ensure matches still make sense.
```bash
uv run python -m scripts.validate_consistency
```
