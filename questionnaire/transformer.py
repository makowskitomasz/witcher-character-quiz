"""Transforms questionnaire answers into normalized feature vectors."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Sequence

import numpy as np

from .schema import Question

TRAIT_ORDER = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "EmotionalStability",
    "Pragmatism",
    "Bravery",
    "MoralFlexibility",
    "Loyalty",
    "Empathy",
    "Independence",
    "Ambition",
    "Adaptability",
    "Impulsiveness",
]


class QuestionnaireTransformer:
    """Converts questionnaire answers to normalized trait feature vectors."""

    def __init__(self, questions: Sequence[Question], trait_order: Sequence[str] | None = None) -> None:
        if not questions:
            raise ValueError("QuestionnaireTransformer requires at least one question.")
        self.questions = list(questions)
        self.question_index: Dict[str, Question] = {question.id: question for question in self.questions}
        if len(self.question_index) != len(self.questions):
            raise ValueError("Duplicate question identifiers detected in transformer initialization.")
        self.trait_order = list(trait_order) if trait_order is not None else list(TRAIT_ORDER)
        self.trait_index = {trait: idx for idx, trait in enumerate(self.trait_order)}

    def transform(self, answers: Mapping[str, str]) -> np.ndarray:
        """Return normalized trait vector for provided answers."""
        if not answers:
            return np.zeros(len(self.trait_order), dtype=float)

        trait_totals = np.zeros(len(self.trait_order), dtype=float)
        trait_counts = defaultdict(int)
        answered_questions = 0

        for question_id, option_id in answers.items():
            question = self._get_question(question_id)
            option = question.get_option(option_id)
            answered_questions += 1
            for trait, value in option.trait_mapping.items():
                idx = self._get_trait_index(trait)
                trait_totals[idx] += value
                trait_counts[idx] += 1

        normalized_vector = np.zeros(len(self.trait_order), dtype=float)
        for idx, total in enumerate(trait_totals):
            count = trait_counts.get(idx, answered_questions)
            if count == 0:
                continue
            normalized_vector[idx] = total / count
        return normalized_vector

    def _get_question(self, question_id: str) -> Question:
        if question_id not in self.question_index:
            raise ValueError(f"Unknown question id '{question_id}'.")
        return self.question_index[question_id]

    def _get_trait_index(self, trait: str) -> int:
        if trait not in self.trait_index:
            raise ValueError(f"Trait '{trait}' is not part of the configured trait_order.")
        return self.trait_index[trait]
