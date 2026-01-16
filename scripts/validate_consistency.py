"""Validates the consistency of character matching across different personas."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from domain.graph_builder import GraphBuilder
from embeddings.hybrid_embeddings import HybridEmbeddings
from embeddings.node2vec_trainer import Node2VecTrainer
from questionnaire.loader import QuestionnaireLoader
from questionnaire.transformer import TRAIT_ORDER, QuestionnaireTransformer
from questionnaire.user_embedding import UserEmbeddingBuilder

def validate_persona(
    name: str,
    answers: Dict[str, str],
    transformer: QuestionnaireTransformer,
    embedding_builder: UserEmbeddingBuilder,
    hybrid: HybridEmbeddings,
    expected_top_names: List[str],
    top_n: int = 5,
) -> bool:
    """Validate a single persona and print results."""
    print(f"\nPersona: {name}")
    print("-" * 20)
    
    feature_vector = transformer.transform(answers)
    user_embedding = embedding_builder.build_from_feature_vector(feature_vector)
    matches = hybrid.match_user_to_character(user_embedding, method="cosine")
    matches = matches[:top_n]
    
    top_match_id, top_similarity = matches[0]
    top_match_name = hybrid.graph.nodes[top_match_id].get("name", top_match_id)
    
    print(f"  Top Match: {top_match_name} ({top_similarity:.1%})")
    print("  Other Matches:")
    for char_id, sim in matches[1:3]:
        name = hybrid.graph.nodes[char_id].get("name", char_id)
        print(f"    - {name} ({sim:.1%})")
        
    is_valid = any(expected in top_match_name for expected in expected_top_names)
    if is_valid:
        print("  ✅ VALID: Met expectations.")
    else:
        print(f"  ❌ INVALID: Expected one of {expected_top_names}")
        
    return is_valid


def main() -> None:
    """Run validation for multiple personas."""
    project_root = Path(__file__).parent.parent
    jsonld_path = project_root / "data" / "raw" / "witcher.jsonld"
    
    # Load and setup
    questions = QuestionnaireLoader.from_default_path().load()
    transformer = QuestionnaireTransformer(questions)
    
    builder = GraphBuilder(jsonld_path, trait_threshold=0.7)
    graph = builder.build()
    
    trainer = Node2VecTrainer(graph, dimensions=32)
    graph_embeddings = trainer.train()
    
    embedding_builder = UserEmbeddingBuilder(graph_embeddings, trait_order=TRAIT_ORDER)
    hybrid = HybridEmbeddings(graph, graph_embeddings)
    hybrid.build_character_embeddings()
    
    # Define personas
    personas = [
        {
            "name": "The Bard (Sociable, Open, Agreeable)",
            "answers": {
                "q1_decision_style": "q1_opt_a",
                "q2_pressure_response": "q2_opt_c",
                "q3_moral_alignment": "q3_opt_c",
                "q4_loyalty": "q4_opt_a",
                "q5_interpersonal": "q5_opt_a",
                "q6_risk": "q6_opt_c",
                "q7_conflict_resolution": "q7_opt_a",
                "q8_adaptability": "q8_opt_a",
                "q9_power_drive": "q9_opt_a",
                "q10_secrecy": "q10_opt_a",
                "q11_magic_use": "q11_opt_a",
                "q12_trust": "q12_opt_a",
            },
            "expected": ["Dandelion"]
        },
        {
            "name": "The Professional (Disciplined, Pragmatic, Resilient)",
            "answers": {
                "q1_decision_style": "q1_opt_b",
                "q2_pressure_response": "q2_opt_a",
                "q3_moral_alignment": "q3_opt_a",
                "q4_loyalty": "q4_opt_a",
                "q5_interpersonal": "q5_opt_b",
                "q6_risk": "q6_opt_b",
                "q7_conflict_resolution": "q7_opt_b",
                "q8_adaptability": "q8_opt_b",
                "q9_power_drive": "q9_opt_b",
                "q10_secrecy": "q10_opt_b",
                "q11_magic_use": "q11_opt_b",
                "q12_trust": "q12_opt_b",
            },
            "expected": ["Geralt", "Vesemir"]
        },
        {
            "name": "The Ambitious (Goal-oriented, Independent, Powerful)",
            "answers": {
                "q1_decision_style": "q1_opt_b",
                "q2_pressure_response": "q2_opt_a",
                "q3_moral_alignment": "q3_opt_b",
                "q4_loyalty": "q4_opt_c",
                "q5_interpersonal": "q5_opt_c",
                "q6_risk": "q6_opt_b",
                "q7_conflict_resolution": "q7_opt_b",
                "q8_adaptability": "q8_opt_c",
                "q9_power_drive": "q9_opt_a",
                "q10_secrecy": "q10_opt_c",
                "q11_magic_use": "q11_opt_a",
                "q12_trust": "q12_opt_b",
            },
            "expected": ["Yennefer", "Triss", "Ciri"]
        }
    ]
    
    print("=" * 60)
    print("Witcher Character Quiz - Consistency Validation")
    print("=" * 60)
    
    all_valid = True
    for p in personas:
        valid = validate_persona(
            p["name"], p["answers"], transformer, embedding_builder, hybrid, p["expected"]
        )
        if not valid:
            all_valid = False
            
    print("\n" + "=" * 60)
    if all_valid:
        print("RESULT: ALL PERSONAS VALID")
    else:
        print("RESULT: SOME PERSONAS FAILED TO MATCH EXPECTATIONS")
    print("=" * 60)


if __name__ == "__main__":
    main()
