"""Configuration for the extraction service."""

from __future__ import annotations

from dataclasses import dataclass

from database.qa_generator import FactKind
from settings import Source, csv, decimal, integer, optional, required

#: The kind the routed extractor always produces. It is not optional: a
#: passage read for nothing else is still read for its claims.
_ALWAYS = FactKind.ATOMIC


def _kinds(name: str, source: Source = None) -> frozenset[str]:
    """Reads which fact kinds this deployment writes.

    Raises:
        KeyError: If the setting is unset or empty.
        ValueError: If it names something that is not a fact kind, or leaves
            out the atomic kind.
    """
    read = frozenset(kind.lower() for kind in csv(name, source))
    unknown = read - set(FactKind)
    if unknown:
        raise ValueError(
            f"{name} names {', '.join(sorted(unknown))}, which is not one of "
            f"{', '.join(FactKind)}"
        )
    if _ALWAYS not in read:
        raise ValueError(f"{name} must name {_ALWAYS}; every passage is read for one")
    return read


def atomic_cap(digests: int, share: float) -> int | None:
    """The most atomic facts one passage may keep, or None for no cap.

    A passage yields a fixed number of digests - one summary, one outline -
    however many claims it carries, so the only thing that moves the balance
    between the kinds is how many atomic facts sit beside them. That makes a
    floor on the other kinds a cap on this one: keeping `n` atomic beside
    `d` others puts the others at d/(n+d), so the largest `n` holding them at
    `share` is d(1-share)/share.

    Measured over this corpus: atomic came back at 18.4 a passage against 2
    digests, which is a 10% share of everything extracted. A third wants 4.
    """
    if digests <= 0 or not 0 < share < 1:
        return None
    return max(1, int(digests * (1 - share) / share))


@dataclass(frozen=True)
class Settings:
    """What extraction writes, and how much of it.

    Attributes:
        kinds: Which fact kinds a run produces.
        digest_share: The longest a summary or an outline may be, as a share
            of the passage it stands in for.
        digest_min_chars: The shortest passage worth digesting at all. 0
            digests every passage that carries enough claims.
        min_other_share: The smallest share of a passage's facts that may be
            something other than atomic. 0 turns the cap off.
        bridges_per_topic: How many bridge calls one topic is worth.
        bridge_passages: How many passages one bridge call is shown.
        model: The model that reads a passage, or None for the shared one.
        digest_model: The model that condenses one, or None to use the same
            one that reads it.
        duplicate_cosine: How alike two statements may be before the second
            is refused. 0 runs no dedup gate and writes no vectors.
        embedding_model: The model the vectors come from.
        embedding_max_tokens: How much of a statement it reads.
    """

    kinds: frozenset[str]
    digest_share: float
    digest_min_chars: int
    min_other_share: float
    bridges_per_topic: int
    bridge_passages: int
    model: str | None
    digest_model: str | None
    duplicate_cosine: float
    #: The cosine two unrelated statements score. Calibration for the stored
    #: confidence only; it decides no verdict.
    duplicate_floor: float
    embedding_model: str
    embedding_max_tokens: int

    @property
    def embeds(self) -> bool:
        """Whether a run embeds what it stores."""
        return self.duplicate_cosine > 0

    @property
    def digests(self) -> tuple[str, ...]:
        """The digest kinds this deployment writes, in storage order."""
        return tuple(
            kind for kind in (FactKind.SUMMARY, FactKind.OUTLINE) if kind in self.kinds
        )

    @property
    def atomic_cap(self) -> int | None:
        """The most atomic facts one passage keeps, or None for no cap."""
        return atomic_cap(len(self.digests), self.min_other_share)

    @property
    def bridging(self) -> bool:
        """Whether a run writes bridge facts at all."""
        return FactKind.BRIDGE in self.kinds

    @classmethod
    def load(cls, source: Source = None) -> Settings:
        """Reads settings from the environment, or from an override.

        Args:
            source: Where to read them, or None for the process environment.
        """
        return cls(
            kinds=_kinds("EXTRACTION_KINDS", source),
            digest_share=decimal("EXTRACTION_DIGEST_MAX_SHARE", source),
            digest_min_chars=integer("EXTRACTION_DIGEST_MIN_CHARS", source),
            min_other_share=decimal("EXTRACTION_MIN_OTHER_SHARE", source),
            bridges_per_topic=integer("EXTRACTION_BRIDGES_PER_TOPIC", source),
            bridge_passages=integer("EXTRACTION_BRIDGE_PASSAGES", source),
            # Absent means the model LLM_MODEL names, which is what every
            # stage called before any of them could name its own.
            model=optional("EXTRACTION_MODEL", source),
            # A digest is the one thing extraction asks for that a small
            # model is actually trained to do, so it is the one worth
            # pointing somewhere cheaper on its own.
            digest_model=optional("EXTRACTION_DIGEST_MODEL", source),
            duplicate_cosine=decimal("EXTRACTION_DUPLICATE_COSINE", source),
            duplicate_floor=decimal("EXTRACTION_DUPLICATE_FLOOR", source),
            embedding_model=required("EMBEDDING_MODEL", source),
            embedding_max_tokens=integer("EMBEDDING_MAX_TOKENS", source),
        )
