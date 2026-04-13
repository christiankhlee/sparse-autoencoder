"""
Factual recall prompt families for sparse feature analysis.

Each family groups paraphrased prompts that share a single correct answer,
plus a small set of "near-miss" or distractor prompts used to study how
feature patterns change when the model is likely to produce incorrect output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class PromptFamily:
    """
    A group of semantically equivalent prompts with a known correct answer.

    Attributes:
        family_id:        Unique string identifier, e.g. ``"capital_france"``.
        topic:            Human-readable topic label.
        correct_answer:   The token (or short string) the model should predict.
        paraphrases:      4-6 prompt strings that all invite the correct answer.
        incorrect_prompts: 2-3 prompts designed to elicit a different or wrong
                           completion (near-misses, misleading phrasing, or a
                           different but related question).
    """

    family_id: str
    topic: str
    correct_answer: str
    paraphrases: List[str] = field(default_factory=list)
    incorrect_prompts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "topic": self.topic,
            "correct_answer": self.correct_answer,
            "paraphrases": self.paraphrases,
            "incorrect_prompts": self.incorrect_prompts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptFamily":
        return cls(
            family_id=d["family_id"],
            topic=d["topic"],
            correct_answer=d["correct_answer"],
            paraphrases=d["paraphrases"],
            incorrect_prompts=d["incorrect_prompts"],
        )


class PromptFamilyBuilder:
    """
    Fluent builder for :class:`PromptFamily` objects.

    Example usage::

        family = (
            PromptFamilyBuilder("capital_france", "Capital of France", "Paris")
            .add_paraphrase("The capital of France is")
            .add_paraphrase("France's capital city is")
            .add_incorrect("The capital of Germany is")
            .build()
        )
    """

    def __init__(self, family_id: str, topic: str, correct_answer: str) -> None:
        self._family_id = family_id
        self._topic = topic
        self._correct_answer = correct_answer
        self._paraphrases: List[str] = []
        self._incorrect_prompts: List[str] = []

    def add_paraphrase(self, prompt: str) -> "PromptFamilyBuilder":
        self._paraphrases.append(prompt)
        return self

    def add_incorrect(self, prompt: str) -> "PromptFamilyBuilder":
        self._incorrect_prompts.append(prompt)
        return self

    def build(self) -> PromptFamily:
        if len(self._paraphrases) < 2:
            raise ValueError(
                f"Family '{self._family_id}' needs at least 2 paraphrases, "
                f"got {len(self._paraphrases)}."
            )
        return PromptFamily(
            family_id=self._family_id,
            topic=self._topic,
            correct_answer=self._correct_answer,
            paraphrases=list(self._paraphrases),
            incorrect_prompts=list(self._incorrect_prompts),
        )


def build_factual_recall_families() -> List[PromptFamily]:
    """
    Return a curated list of factual-recall prompt families.

    Covers three categories:
    - Capital cities
    - Famous inventors
    - Country / world facts

    Returns:
        List of :class:`PromptFamily` instances.
    """
    families: List[PromptFamily] = []

    # ------------------------------------------------------------------
    # Capital cities
    # ------------------------------------------------------------------

    families.append(
        PromptFamilyBuilder("capital_france", "Capital of France", " Paris")
        .add_paraphrase("The capital of France is")
        .add_paraphrase("France's capital city is")
        .add_paraphrase("The capital city of France is")
        .add_paraphrase("What is the capital of France? The answer is")
        .add_paraphrase("In France, the capital is")
        .add_incorrect("The largest city in France is")  # answer is also Paris but different framing
        .add_incorrect("The capital of Germany is")  # different country
        .build()
    )

    families.append(
        PromptFamilyBuilder("capital_japan", "Capital of Japan", " Tokyo")
        .add_paraphrase("The capital of Japan is")
        .add_paraphrase("Japan's capital city is")
        .add_paraphrase("The capital city of Japan is")
        .add_paraphrase("What is the capital of Japan? The answer is")
        .add_paraphrase("In Japan, the seat of government is located in")
        .add_incorrect("The largest city in China is")  # different country
        .add_incorrect("The former capital of Japan was")  # answer: Kyoto
        .build()
    )

    families.append(
        PromptFamilyBuilder("capital_germany", "Capital of Germany", " Berlin")
        .add_paraphrase("The capital of Germany is")
        .add_paraphrase("Germany's capital city is")
        .add_paraphrase("The capital city of Germany is")
        .add_paraphrase("What is the capital of Germany? The answer is")
        .add_paraphrase("In Germany, the federal capital is")
        .add_incorrect("The capital of Austria is")  # different but related country
        .add_incorrect("The largest city in Germany is")  # same answer but different framing
        .build()
    )

    families.append(
        PromptFamilyBuilder("capital_italy", "Capital of Italy", " Rome")
        .add_paraphrase("The capital of Italy is")
        .add_paraphrase("Italy's capital city is")
        .add_paraphrase("The capital city of Italy is")
        .add_paraphrase("What is the capital of Italy? The answer is")
        .add_paraphrase("In Italy, the capital and largest city is")
        .add_incorrect("The capital of Spain is")  # different country
        .add_incorrect("The ancient capital of the Roman Empire was")  # historical framing
        .build()
    )

    families.append(
        PromptFamilyBuilder("capital_spain", "Capital of Spain", " Madrid")
        .add_paraphrase("The capital of Spain is")
        .add_paraphrase("Spain's capital city is")
        .add_paraphrase("The capital city of Spain is")
        .add_paraphrase("What is the capital of Spain? The answer is")
        .add_paraphrase("In Spain, the capital is")
        .add_incorrect("The capital of Portugal is")  # neighboring country
        .add_incorrect("The most populous city in Spain is")  # same answer, different framing
        .build()
    )

    families.append(
        PromptFamilyBuilder("capital_australia", "Capital of Australia", " Canberra")
        .add_paraphrase("The capital of Australia is")
        .add_paraphrase("Australia's capital city is")
        .add_paraphrase("The capital city of Australia is")
        .add_paraphrase("What is the capital of Australia? The answer is")
        .add_paraphrase("In Australia, the seat of government is")
        .add_incorrect("The largest city in Australia is")  # Sydney — common confusion
        .add_incorrect("The capital of New Zealand is")  # different country
        .build()
    )

    families.append(
        PromptFamilyBuilder("capital_brazil", "Capital of Brazil", " Bras")
        .add_paraphrase("The capital of Brazil is")
        .add_paraphrase("Brazil's capital city is")
        .add_paraphrase("The capital city of Brazil is")
        .add_paraphrase("What is the capital of Brazil? The answer is")
        .add_paraphrase("In Brazil, the federal capital is")
        .add_incorrect("The largest city in Brazil is")  # Sao Paulo — common confusion
        .add_incorrect("The capital of Argentina is")  # different country
        .build()
    )

    families.append(
        PromptFamilyBuilder("capital_canada", "Capital of Canada", " Ottawa")
        .add_paraphrase("The capital of Canada is")
        .add_paraphrase("Canada's capital city is")
        .add_paraphrase("The capital city of Canada is")
        .add_paraphrase("What is the capital of Canada? The answer is")
        .add_paraphrase("In Canada, the national capital is")
        .add_incorrect("The largest city in Canada is")  # Toronto — common confusion
        .add_incorrect("The capital of the United States is")  # neighboring country
        .build()
    )

    # ------------------------------------------------------------------
    # Famous inventors
    # ------------------------------------------------------------------

    families.append(
        PromptFamilyBuilder("inventor_telephone", "Inventor of the Telephone", " Bell")
        .add_paraphrase("The inventor of the telephone was Alexander Graham")
        .add_paraphrase("The telephone was invented by Alexander Graham")
        .add_paraphrase("Alexander Graham")
        .add_paraphrase("Who invented the telephone? The inventor was Alexander Graham")
        .add_paraphrase("The person credited with inventing the telephone is Alexander Graham")
        .add_incorrect("The inventor of the radio was")  # Marconi
        .add_incorrect("The inventor of the telegraph was")  # Morse
        .build()
    )

    families.append(
        PromptFamilyBuilder("inventor_lightbulb", "Inventor of the Lightbulb", " Edison")
        .add_paraphrase("The inventor of the lightbulb was Thomas")
        .add_paraphrase("The lightbulb was invented by Thomas")
        .add_paraphrase("Thomas Alva")
        .add_paraphrase("Who invented the practical lightbulb? The inventor was Thomas")
        .add_paraphrase("The person credited with inventing the lightbulb is Thomas")
        .add_incorrect("The inventor of the telephone was Thomas")  # wrong attribution
        .add_incorrect("The inventor of alternating current was")  # Tesla
        .build()
    )

    families.append(
        PromptFamilyBuilder("inventor_gravity", "Discoverer of Gravity", " Newton")
        .add_paraphrase("The scientist who formulated the law of gravity was Isaac")
        .add_paraphrase("Gravity was described mathematically by Isaac")
        .add_paraphrase("The theory of universal gravitation was developed by Isaac")
        .add_paraphrase("Who discovered gravity? The answer is Isaac")
        .add_paraphrase("The famous apple story is associated with Isaac")
        .add_incorrect("The scientist who developed the theory of relativity was")  # Einstein
        .add_incorrect("The discoverer of evolution was")  # Darwin
        .build()
    )

    families.append(
        PromptFamilyBuilder("inventor_relativity", "Developer of Relativity", " Einstein")
        .add_paraphrase("The theory of relativity was developed by Albert")
        .add_paraphrase("The renowned physicist Albert")
        .add_paraphrase("The scientist famous for the equation E equals mc squared is Albert")
        .add_paraphrase("Who developed the special theory of relativity? The answer is Albert")
        .add_paraphrase("The physicist who revolutionized our understanding of space-time was Albert")
        .add_incorrect("The theory of gravity was formulated by Albert")  # Newton
        .add_incorrect("The inventor of the lightbulb was Albert")  # Edison
        .build()
    )

    families.append(
        PromptFamilyBuilder("inventor_evolution", "Developer of Evolution Theory", " Darwin")
        .add_paraphrase("The theory of evolution by natural selection was proposed by Charles")
        .add_paraphrase("On the Origin of Species was written by Charles")
        .add_paraphrase("The naturalist who sailed on HMS Beagle was Charles")
        .add_paraphrase("Who proposed the theory of natural selection? The answer is Charles")
        .add_paraphrase("The scientist associated with evolution and the Galapagos Islands is Charles")
        .add_incorrect("The scientist who developed the theory of relativity was Charles")  # Einstein
        .add_incorrect("The inventor of the telephone was Charles")  # Bell
        .build()
    )

    # ------------------------------------------------------------------
    # Country / world facts
    # ------------------------------------------------------------------

    families.append(
        PromptFamilyBuilder("largest_country", "Largest Country by Area", " Russia")
        .add_paraphrase("The largest country in the world by land area is")
        .add_paraphrase("By total area, the biggest country on Earth is")
        .add_paraphrase("The world's largest nation by territory is")
        .add_paraphrase("Which country has the most land area? The answer is")
        .add_paraphrase("In terms of geographic size, the largest country is")
        .add_incorrect("The most populous country in the world is")  # China / India
        .add_incorrect("The largest country in North America is")  # Canada
        .build()
    )

    families.append(
        PromptFamilyBuilder("longest_river", "Longest River in the World", " Nile")
        .add_paraphrase("The longest river in the world is the")
        .add_paraphrase("By length, the world's longest river is the")
        .add_paraphrase("The river that holds the record for being the longest on Earth is the")
        .add_paraphrase("Which river is considered the longest in the world? The answer is the")
        .add_paraphrase("The most famous river in Africa, also the world's longest, is the")
        .add_incorrect("The longest river in South America is the")  # Amazon
        .add_incorrect("The longest river in Europe is the")  # Volga
        .build()
    )

    families.append(
        PromptFamilyBuilder("highest_mountain", "Highest Mountain on Earth", " Everest")
        .add_paraphrase("The highest mountain in the world is Mount")
        .add_paraphrase("The tallest mountain on Earth is Mount")
        .add_paraphrase("The tallest peak on Earth above sea level is Mount")
        .add_paraphrase("Which mountain is the highest in the world? The answer is Mount")
        .add_paraphrase("The mountain located in the Himalayas that is the world's highest is Mount")
        .add_incorrect("The highest mountain in Africa is Mount")  # Kilimanjaro
        .add_incorrect("The highest mountain in Europe is Mount")  # Elbrus / Mont Blanc
        .build()
    )

    logger.info("Built %d prompt families.", len(families))
    return families
