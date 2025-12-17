"""Utilities for loading and validating questionnaire definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .schema import Question, Questionnaire


class QuestionnaireLoader:
    """Loader responsible for ingesting questionnaire definitions from JSON."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> List[Question]:
        payload = self._read_json()
        questionnaire = Questionnaire.model_validate(payload)
        return list(questionnaire.questions)

    def _read_json(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"Questionnaire file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    @classmethod
    def from_default_path(
        cls, base_dir: Path | str | None = None
    ) -> "QuestionnaireLoader":
        base = (
            Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
        )
        return cls(base / "questionnaire.json")
