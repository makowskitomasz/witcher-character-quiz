"""Utilities for constructing the knowledge graph from domain data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from questionnaire.transformer import TRAIT_ORDER
from utils.io import (
    character_trait_vector,
    extract_characters,
    extract_factions,
    extract_traits,
    load_jsonld,
)


class GraphBuilder:
    """Constructs NetworkX graph from Witcher domain data."""

    def __init__(
        self,
        jsonld_path: Path | str,
        trait_threshold: float = 0.7,
        trait_prefix: str = "wc:",
    ) -> None:
        """Initialize graph builder.
        
        Args:
            jsonld_path: Path to JSON-LD data file
            trait_threshold: Minimum trait value to create hasHighTrait edge
            trait_prefix: Prefix used for trait IDs in JSON-LD
        """
        self.jsonld_path = Path(jsonld_path)
        self.trait_threshold = trait_threshold
        self.trait_prefix = trait_prefix
        self.graph: nx.Graph | None = None
        self.data: Dict[str, Any] | None = None

    def build(self) -> nx.Graph:
        """Construct the domain graph representation.
        
        Returns:
            NetworkX graph with nodes for Characters, Factions, and Traits
        """
        # Load data
        self.data = load_jsonld(self.jsonld_path)
        
        # Create undirected graph
        self.graph = nx.Graph()
        
        # Extract entities
        characters = extract_characters(self.data)
        factions = extract_factions(self.data)
        traits = extract_traits(self.data)
        
        # Create node mapping for easy lookup
        trait_id_to_name = {}
        for trait in traits:
            trait_id = trait.get("id", "")
            trait_name = trait.get("name", "")
            trait_id_to_name[trait_id] = trait_name
        
        faction_id_to_name = {}
        for faction in factions:
            faction_id = faction.get("id", "")
            faction_name = faction.get("name", "")
            faction_id_to_name[faction_id] = faction_name
        
        # Add nodes with attributes
        for character in characters:
            char_id = character.get("id", "")
            char_name = character.get("name", "")
            self.graph.add_node(
                char_id,
                type="Character",
                name=char_name,
                trait_vector=character_trait_vector(character, TRAIT_ORDER, self.trait_prefix),
            )
        
        for faction in factions:
            faction_id = faction.get("id", "")
            faction_name = faction.get("name", "")
            self.graph.add_node(faction_id, type="Faction", name=faction_name)
        
        for trait in traits:
            trait_id = trait.get("id", "")
            trait_name = trait.get("name", "")
            self.graph.add_node(trait_id, type="TraitDimension", name=trait_name)
        
        # Add edges
        # 1. Character --hasFaction--> Faction
        for character in characters:
            char_id = character.get("id", "")
            char_factions = character.get("hasFaction", [])
            for faction_id in char_factions:
                if faction_id in self.graph:
                    self.graph.add_edge(char_id, faction_id, relation_type="hasFaction")
        
        # 2. Character --relation--> Character
        for character in characters:
            char_id = character.get("id", "")
            relations = character.get("relation", [])
            for rel in relations:
                target_id = rel.get("target", "")
                rel_type = rel.get("type", "")
                if target_id in self.graph:
                    self.graph.add_edge(char_id, target_id, relation_type=rel_type)
        
        # 3. Character --hasHighTrait--> TraitDimension (if trait >= threshold)
        for character in characters:
            char_id = character.get("id", "")
            trait_values = character.get("traitValues", {})
            for trait_id, value in trait_values.items():
                if isinstance(value, (int, float)) and value >= self.trait_threshold:
                    if trait_id in self.graph:
                        self.graph.add_edge(char_id, trait_id, relation_type="hasHighTrait", weight=value)
        
        return self.graph

    def get_character_nodes(self) -> List[str]:
        """Get list of character node IDs."""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build() first.")
        return [n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "Character"]

    def get_faction_nodes(self) -> List[str]:
        """Get list of faction node IDs."""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build() first.")
        return [n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "Faction"]

    def get_trait_nodes(self) -> List[str]:
        """Get list of trait node IDs."""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build() first.")
        return [n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "TraitDimension"]
