"""Generates natural language explanations for character matches."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class ResultExplainer:
    """Compares user and character traits to explain why they match."""

    def __init__(self, trait_order: Sequence[str]) -> None:
        """Initialize with the order of traits in the vectors.

        Args:
            trait_order: List of trait names corresponding to vector indices.
        """
        self.trait_order = trait_order

    def explain(
        self,
        user_vector: np.ndarray,
        character_vector: np.ndarray,
        character_name: str,
        top_n: int = 3,
    ) -> str:
        """Generate an explanation for why the user matches the character.

        Args:
            user_vector: User's trait feature vector (14D)
            character_vector: Character's trait feature vector (14D)
            character_name: Name of the matched character
            top_n: Number of key traits to include in the explanation

        Returns:
            A descriptive string explanation.
        """
        # Ensure we only use the trait part of the vectors if they are hybrid
        user_traits = user_vector[: len(self.trait_order)]
        char_traits = character_vector[: len(self.trait_order)]

        # Calculate "alignment" score for each trait
        # We look for traits where both have high values or both have low values
        # (1 - |user - char|) gives a high score for similarity
        similarities = 1.0 - np.abs(user_traits - char_traits)

        # We also want to prioritize traits that are "pronounced" (either high or low)
        # because those define the character more than "average" traits.
        importance = np.abs(char_traits - 0.5) * 2.0  # 0.0 at center, 1.0 at edges

        combined_score = similarities * (1.0 + importance)

        # Get indices of top traits
        top_indices = np.argsort(combined_score)[-top_n:][::-1]

        reasons = []
        for idx in top_indices:
            trait_name = self.trait_order[idx]
            user_val = user_traits[idx]
            char_val = char_traits[idx]

            if char_val > 0.7 and user_val > 0.6:
                reasons.append(f"your strong {trait_name.lower()}")
            elif char_val < 0.3 and user_val < 0.4:
                reasons.append(f"your shared lack of {trait_name.lower()}")
            else:
                reasons.append(f"your similar level of {trait_name.lower()}")

        if not reasons:
            return f"You are a great match for {character_name} because your overall personality profile aligns closely with theirs."

        if len(reasons) == 1:
            explanation = f"You are most like {character_name} primarily because of {reasons[0]}."
        elif len(reasons) == 2:
            explanation = f"You are most like {character_name} because of {reasons[0]} and {reasons[1]}."
        else:
            explanation = (
                f"You are most like {character_name} because of {', '.join(reasons[:-1])}, "
                f"and {reasons[-1]}."
            )

        return explanation
