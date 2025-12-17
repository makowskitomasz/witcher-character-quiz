"""IO utilities for loading domain data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_jsonld(file_path: Path | str) -> Dict[str, Any]:
    """Load JSON-LD file and return parsed dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON-LD file not found: {path}")
    with path.open() as f:
        return json.load(f)


def extract_characters(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all Character entities from JSON-LD graph."""
    characters = []
    graph = data.get("@graph", [])
    for item in graph:
        if item.get("@type") == "Character":
            characters.append(item)
    return characters


def extract_factions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all Faction entities from JSON-LD graph."""
    factions = []
    graph = data.get("@graph", [])
    for item in graph:
        if item.get("@type") == "Faction":
            factions.append(item)
    return factions


def extract_traits(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all TraitDimension entities from JSON-LD graph."""
    traits = []
    graph = data.get("@graph", [])
    for item in graph:
        if item.get("@type") == "TraitDimension":
            traits.append(item)
    return traits


def character_trait_vector(
    character: Dict[str, Any], trait_order: List[str], trait_prefix: str = "wc:"
) -> np.ndarray:
    """Extract trait vector for a character as numpy array.

    Args:
        character: Character dictionary from JSON-LD
        trait_order: Ordered list of trait names (without prefix)
        trait_prefix: Prefix used in JSON-LD (default: "wc:")

    Returns:
        numpy array of shape (len(trait_order),) with trait values
    """
    trait_values = character.get("traitValues", {})
    vector = np.zeros(len(trait_order), dtype=float)

    for idx, trait_name in enumerate(trait_order):
        trait_key = f"{trait_prefix}{trait_name}"
        if trait_key in trait_values:
            vector[idx] = float(trait_values[trait_key])

    return vector
