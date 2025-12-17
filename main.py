"""Simple CLI runner for the Witcher questionnaire pipeline with character matching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from domain.graph_builder import GraphBuilder
from embeddings.hybrid_embeddings import HybridEmbeddings
from embeddings.node2vec_trainer import Node2VecTrainer
from questionnaire.loader import QuestionnaireLoader
from questionnaire.schema import Option, Question
from questionnaire.transformer import TRAIT_ORDER, QuestionnaireTransformer
from questionnaire.user_embedding import UserEmbeddingBuilder


class QuestionnaireApp:
    """Guides the user through the questionnaire and matches to characters."""

    def __init__(self, jsonld_path: Path | str | None = None) -> None:
        """Initialize the questionnaire app.

        Args:
            jsonld_path: Path to JSON-LD file (defaults to data/raw/witcher.jsonld)
        """
        self.questions = QuestionnaireLoader.from_default_path().load()
        self.transformer = QuestionnaireTransformer(self.questions)

        # Set up embeddings
        if jsonld_path is None:
            project_root = Path(__file__).parent
            jsonld_path = project_root / "data" / "raw" / "witcher.jsonld"

        print("Loading character data and training embeddings...")
        self.builder = GraphBuilder(jsonld_path, trait_threshold=0.7)
        self.graph = self.builder.build()

        print("  Training graph embeddings...")
        self.trainer = Node2VecTrainer(self.graph, dimensions=32)
        self.graph_embeddings = self.trainer.train()

        self.user_embedding_builder = UserEmbeddingBuilder(
            self.graph_embeddings, trait_order=TRAIT_ORDER
        )

        print("  Building hybrid embeddings...")
        self.hybrid = HybridEmbeddings(self.graph, self.graph_embeddings)
        self.hybrid.build_character_embeddings()

        print("Ready!\n")

    def run(self) -> None:
        """Run the questionnaire and match to characters."""
        answers = self._collect_answers()
        feature_vector = self.transformer.transform(answers)
        user_embedding = self.user_embedding_builder.build_from_feature_vector(
            feature_vector
        )
        matches = self.hybrid.match_user_to_character(user_embedding, method="cosine")

        self._print_vector(feature_vector)
        self._print_matches(matches)

    def _collect_answers(self) -> Dict[str, str]:
        """Collect answers from user."""
        answers: Dict[str, str] = {}
        for question in self.questions:
            option = self._prompt_for_option(question)
            answers[question.id] = option.id
        return answers

    def _prompt_for_option(self, question: Question) -> Option:
        """Prompt user for an option."""
        print(f"\n{question.text}")
        for idx, option in enumerate(question.options, start=1):
            print(f"  [{idx}] {option.text}")
        while True:
            raw = input("Select option by number: ").strip()
            if not raw:
                continue
            try:
                idx = int(raw)
                if 1 <= idx <= len(question.options):
                    return question.options[idx - 1]
            except ValueError:
                pass
            print("Invalid selection. Please enter a number.")

    def _print_vector(self, vector) -> None:
        """Print the trait vector."""
        print("\n" + "=" * 60)
        print("Your Trait Profile:")
        print("=" * 60)
        for trait, value in zip(TRAIT_ORDER, vector):
            bar_length = int(value * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {trait:>20}: {value:.3f} [{bar}]")

    def _print_matches(self, matches: list[tuple[str, float]]) -> None:
        """Print character matches."""
        print("\n" + "=" * 60)
        print("Character Matches:")
        print("=" * 60)
        for rank, (char_id, similarity) in enumerate(matches, start=1):
            node_data = self.graph.nodes[char_id]
            char_name = node_data.get("name", char_id)
            percentage = similarity * 100
            bar_length = int(similarity * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"\n  #{rank}: {char_name}")
            print(f"  Match: {percentage:5.1f}% [{bar}]")

            # Show faction
            factions = []
            for neighbor in self.graph.neighbors(char_id):
                neighbor_data = self.graph.nodes.get(neighbor, {})
                if neighbor_data.get("type") == "Faction":
                    factions.append(neighbor_data.get("name", neighbor))
            if factions:
                print(f"  Faction: {', '.join(factions)}")


if __name__ == "__main__":
    app = QuestionnaireApp()
    app.run()
