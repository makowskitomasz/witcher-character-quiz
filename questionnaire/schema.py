"""Pydantic models describing the Witcher questionnaire."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

TraitMapping = Dict[str, float]


class Option(BaseModel):
    """Single answer option and its contribution to trait scores."""

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    trait_mapping: TraitMapping = Field(..., alias="trait_mapping")

    @field_validator("trait_mapping")
    def validate_trait_mapping(cls, mapping: TraitMapping) -> TraitMapping:
        if not mapping:
            raise ValueError("trait_mapping must contain at least one trait weight.")
        for trait, score in mapping.items():
            if not isinstance(trait, str) or not trait:
                raise ValueError("trait_mapping keys must be non-empty strings.")
            if not isinstance(score, (int, float)):
                raise ValueError("trait_mapping values must be numeric.")
            if score < 0.0 or score > 1.0:
                raise ValueError("trait_mapping values must be within [0.0, 1.0].")
        return mapping


class Question(BaseModel):
    """Question definition consisting of text and option list."""

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    options: List[Option] = Field(..., min_length=2)

    @field_validator("options")
    def validate_unique_option_ids(cls, options: List[Option]) -> List[Option]:
        seen = set()
        for option in options:
            if option.id in seen:
                raise ValueError(f"Duplicate option id detected: {option.id}")
            seen.add(option.id)
        return options

    def get_option(self, option_id: str) -> Option:
        for option in self.options:
            if option.id == option_id:
                return option
        raise ValueError(
            f"Option id '{option_id}' does not exist for question '{self.id}'."
        )


class Questionnaire(BaseModel):
    """Root questionnaire container."""

    questions: List[Question] = Field(..., min_length=1)

    @field_validator("questions")
    def validate_unique_question_ids(cls, questions: List[Question]) -> List[Question]:
        seen = set()
        for question in questions:
            if question.id in seen:
                raise ValueError(f"Duplicate question id detected: {question.id}")
            seen.add(question.id)
        return questions
