"""The things topic modelling passes around."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PassageVocabulary:
    """One passage as the fitter receives it: its id and its lemmas."""

    id: int
    lemmas: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PassageWeight:
    """How strongly one passage belongs to one topic."""

    passage_id: int
    #: Position in the fitted model, not the topics table's primary key.
    topic_index: int
    weight: float


@dataclass(frozen=True)
class TopicSpace:
    """One language's fitted model, as pyLDAvis reads it."""

    #: Term weight per topic, one row per topic, summing to 1.
    topic_term: list[list[float]]
    #: Topic weight per passage, one row per passage, summing to 1.
    doc_topic: list[list[float]]
    #: Term occurrences in each passage of `doc_topic`, in the same order.
    doc_lengths: list[int]
    #: The vocabulary, in the column order of `topic_term`.
    vocabulary: list[str]
    #: Occurrences of each term across the corpus, in the same order.
    term_frequency: list[int]


@dataclass(frozen=True)
class FittedTopic:
    """One topic as the model produced it."""

    topic_index: int
    top_terms: list[str]
    label: str | None = None
    #: What named it: "person", or the model identifier.
    labelled_by: str | None = None
    include_in_coverage: bool = True


@dataclass(frozen=True)
class Fitting:
    """What one run made of one language's passages."""

    language: str
    topics: list[FittedTopic]
    #: Every membership above the weight floor.
    weights: list[PassageWeight]
    #: How many passages were fitted over.
    passages: int
    #: How many of them ended up in no topic.
    without_topics: int
    #: How many terms survived the frequency filter.
    vocabulary: int
    #: The whole fitted model, for the visualisation.
    space: TopicSpace

    @property
    def labelled(self) -> int:
        """How many of these topics hold a label, whatever named them."""
        return sum(1 for topic in self.topics if topic.label)


@dataclass(frozen=True)
class StoredTopic:
    """One topic as it is read back out, with how much of the corpus it holds."""

    id: int
    language: str | None
    topic_index: int
    top_terms: list[str]
    label: str | None
    labelled_by: str | None
    include_in_coverage: bool
    #: How many passages hold it with any weight.
    passages: int
    #: How many passages hold it as their highest weight.
    dominant_passages: int
    #: Mean weight across the passages holding it.
    mean_weight: float
    #: How many documents those passages come from.
    documents: int
    #: How many of the passages it is dominant in are tables rather than prose.
    table_passages: int
    #: Validated facts drawn from the passages it is dominant in.
    validated_facts: int


@dataclass(frozen=True)
class LanguageFit:
    """What one language's model was fitted over, and whether it still holds."""

    language: str
    topics: int
    corpus_passages: int | None
    corpus_vocabulary: int | None
    passages_without_topics: int | None
    fitted_at: str | None
    #: Passages of this language the corpus holds now, and memberships its
    #: topics hold now.
    live_passages: int = 0
    memberships: int = 0

    @property
    def stale(self) -> bool:
        """Whether this language holds topics that no longer fit the corpus.

        False when it holds none at all.
        """
        if not self.topics:
            return False
        return self.memberships == 0 or self.corpus_passages != self.live_passages


@dataclass(frozen=True)
class TopicFit:
    """The current topic model, and the state of any fit asked for."""

    status: str | None
    error: str | None
    requested_at: str | None
    topics: int
    languages: list[LanguageFit] = field(default_factory=list)
    #: Passages carrying no language at all, which no model covers.
    passages_without_language: int = 0

    @property
    def stale(self) -> bool:
        """Whether any language's topics no longer fit the corpus."""
        return any(fit.stale for fit in self.languages)


@dataclass(frozen=True)
class TopicRemoval:
    """What deleting the topics took with it."""

    topics: int
    memberships: int
    #: How many removed topics carried a label, whatever named them.
    labels: int = 0
    #: The languages whose topics went, for the visualisations keyed by one.
    languages: list[str] = field(default_factory=list)
