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

## Project Structure
This milestone focuses on the skeleton; implementation details will arrive with future milestones.

## How to Extend the Questionnaire
- Update `questionnaire/questionnaire.json` by appending new questions that follow the existing schema (unique IDs, text, options with `trait_mapping` weights). Keep trait names aligned with the canonical list in `questionnaire/transformer.py`.
- Run `QuestionnaireLoader.from_default_path().load()` to validate the JSON through the Pydantic models in `questionnaire/schema.py`. Validation errors point directly to malformed entries.
- If you introduce new trait dimensions, expand `TRAIT_ORDER` inside `questionnaire/transformer.py` to keep feature vectors aligned and reproducible across experiments.
- Extend transformation behavior by subclassing or modifying `QuestionnaireTransformer`—for example to weight certain questions higher or to output additional metadata alongside the normalized vector.
