"""Simple CLI runner for the Witcher questionnaire pipeline."""

from __future__ import annotations

from typing import Dict

from questionnaire.loader import QuestionnaireLoader
from questionnaire.schema import Question, Option
from questionnaire.transformer import QuestionnaireTransformer, TRAIT_ORDER


class QuestionnaireApp:
    """Guides the user through the questionnaire and prints a trait vector."""

    def __init__(self) -> None:
        self.questions = QuestionnaireLoader.from_default_path().load()
        self.transformer = QuestionnaireTransformer(self.questions)

    def run(self) -> None:
        answers = self._collect_answers()
        feature_vector = self.transformer.transform(answers)
        self._print_vector(feature_vector)

    def _collect_answers(self) -> Dict[str, str]:
        answers: Dict[str, str] = {}
        for question in self.questions:
            option = self._prompt_for_option(question)
            answers[question.id] = option.id
        return answers

    def _prompt_for_option(self, question: Question) -> Option:
        print(f"\n{question.text}")
        for idx, option in enumerate(question.options, start=1):
            print(f"  [{idx}] {option.text} (id: {option.id})")
        while True:
            raw = input("Select option by number or id: ").strip()
            if not raw:
                continue
            try:
                idx = int(raw)
                if 1 <= idx <= len(question.options):
                    return question.options[idx - 1]
            except ValueError:
                pass
            try:
                return question.get_option(raw)
            except ValueError as exc:
                print(exc)

    def _print_vector(self, vector) -> None:
        print("\nDerived trait vector:")
        for trait, value in zip(TRAIT_ORDER, vector):
            print(f"  {trait:>20}: {value:.3f}")


app = QuestionnaireApp()
app.run()
