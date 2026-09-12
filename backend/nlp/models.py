"""The things the nlp package hands back.

Plain values, and no spaCy import: a caller that only needs the shape of a
sentence should not load a pipeline to get it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    """One sentence of a passage, located in the passage text."""

    index: int
    start: int
    end: int
    text: str
    predicates: int


@dataclass(frozen=True)
class Claim:
    """What a statement asserts, and what it leaves unsaid."""

    predicates: int
    units: tuple[str, ...]
    references: tuple[str, ...]
