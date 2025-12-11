.PHONY: run build_graph export_embeddings visualize_graph format test

run:
	uv run python main.py

build_graph:
	uv run python -m scripts.build_graph

export_embeddings:
	uv run python -m scripts.export_embeddings

visualize_graph:
	uv run python -m scripts.visualize_graph

format:
	uv run black .
	uv run isort .

test:
	uv run pytest
