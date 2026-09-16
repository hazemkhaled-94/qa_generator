"""Configuration for the extraction service."""

from __future__ import annotations

from dataclasses import dataclass

from database.qa_generator import FactKind
from settings import csv, decimal, integer

#: The kind the routed extractor always produces. It is not optional: a
#: passage read for nothing else is still read for its claims.
_ALWAYS = FactKind.ATOMIC


def _kinds(name: str) -> frozenset[str]:
    """Reads which fact kinds this deployment writes.

    Raises:
        KeyError: If the setting is unset or empty.
        ValueError: If it names something that is not a fact kind, or leaves
            out the atomic kind.
    """
    read = frozenset(kind.lower() for kind in csv(name))
    unknown = read - set(FactKind)
    if unknown:
        raise ValueError(
            f"{name} names {', '.join(sorted(unknown))}, which is not one of "
            f"{', '.join(FactKind)}"
        )
    if _ALWAYS not in read:
        raise ValueError(f"{name} must name {_ALWAYS}; every passage is read for one")
    return read


@dataclass(frozen=True)
class Settings:
    """What extraction writes, and how much of it.

    Attributes:
        kinds: Which fact kinds a run produces.
        digest_share: The longest a summary or an outline may be, as a share
            of the passage it stands in for.
        bridges_per_topic: How many bridge calls one topic is worth.
        bridge_passages: How many passages one bridge call is shown.
    """

    kinds: frozenset[str]
    digest_share: float
    bridges_per_topic: int
    bridge_passages: int

    @property
    def digests(self) -> tuple[str, ...]:
        """The digest kinds this deployment writes, in storage order."""
        return tuple(
            kind for kind in (FactKind.SUMMARY, FactKind.OUTLINE) if kind in self.kinds
        )

    @property
    def bridging(self) -> bool:
        """Whether a run writes bridge facts at all."""
        return FactKind.BRIDGE in self.kinds

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment."""
        return cls(
            kinds=_kinds("EXTRACTION_KINDS"),
            digest_share=decimal("EXTRACTION_DIGEST_MAX_SHARE"),
            bridges_per_topic=integer("EXTRACTION_BRIDGES_PER_TOPIC"),
            bridge_passages=integer("EXTRACTION_BRIDGE_PASSAGES"),
        )
