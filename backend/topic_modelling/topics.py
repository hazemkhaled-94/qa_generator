"""Fitting a topic model over one language's vocabulary."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from itertools import batched
from typing import cast

from gensim import corpora
from gensim.models import Nmf, TfidfModel

from topic_modelling.models import (
    FittedTopic,
    Fitting,
    PassageVocabulary,
    PassageWeight,
    TopicSpace,
)

log = logging.getLogger(__name__)

#: How many passages to hold at once while building the vocabulary.
_BATCH = 500

#: The corpus, as the fitter takes it: a callable handing back a fresh walk
#: over every passage.
Corpus = Callable[[], Iterable[PassageVocabulary]]

#: Smallest share of terms two topics must have in common before a label is
#: carried from one to the other.
_LABEL_MATCH = 0.4


class NoVocabulary(Exception):
    """Raised when there is nothing to fit a model on."""


class _BagsOfWords:
    """The corpus as bags of words, rebuilt on every walk."""

    def __init__(self, corpus: Corpus, dictionary: corpora.Dictionary) -> None:
        """Initialises the view."""
        self._corpus = corpus
        self._dictionary = dictionary

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        """Walks the corpus once."""
        for passage in self._corpus():
            yield cast(
                "list[tuple[int, int]]", self._dictionary.doc2bow(passage.lemmas)
            )


class TopicFitter:
    """Fits topics over the terms one language's passages use.

    Non-negative matrix factorisation over a tf-idf weighted space. The
    vocabulary is the content lemmas chunking stored. The weights returned
    are normalised, and the fit is seeded.
    """

    def __init__(
        self,
        *,
        num_topics: int,
        passes: int,
        random_state: int,
        top_terms: int,
        min_weight: float,
        no_below: int,
        no_above: float,
    ) -> None:
        """Initialises the fitter.

        Args:
            num_topics: Topics to fit, at least 2.
            passes: Times the factorisation walks the corpus.
            random_state: Seed for the factorisation.
            top_terms: Terms kept as a topic's signature, at least 1.
            min_weight: Smallest membership weight kept, in (0, 1].
            no_below: Fewest passages a term must appear in.
            no_above: Largest share of passages a term may appear in, in
                (0, 1].

        Raises:
            ValueError: If a setting is outside the range named above.
        """
        if num_topics < 2:
            raise ValueError(
                f"TOPIC_NUM_TOPICS={num_topics} is not a partition of anything; "
                "it must be at least 2"
            )
        if top_terms < 1:
            raise ValueError(
                f"TOPIC_TOP_TERMS={top_terms} leaves a topic with no signature; "
                "it must be at least 1"
            )
        if not 0 < min_weight <= 1:
            raise ValueError(
                f"TOPIC_MIN_WEIGHT={min_weight} is not a probability floor; it "
                "must be above 0 and at most 1"
            )
        if not 0 < no_above <= 1:
            raise ValueError(
                f"TOPIC_NO_ABOVE={no_above} is not a share of the corpus; it must "
                "be above 0 and at most 1"
            )
        self._num_topics = num_topics
        self._passes = passes
        self._random_state = random_state
        self._top_terms = top_terms
        self._min_weight = min_weight
        self._no_below = no_below
        self._no_above = no_above

    def fit(self, corpus: Corpus, language: str) -> Fitting:
        """Fits one language's model and scores its passages against it.

        The corpus is walked rather than held: once to build the vocabulary,
        once per pass, and once more to score.

        Args:
            corpus: Hands back a fresh walk over the passages each time it is
                called.
            language: The ISO 639-1 code the fit is recorded against.

        Returns:
            The topics, the memberships above the weight floor, and the whole
            fitted space.

        Raises:
            NoVocabulary: If the corpus is empty, if the frequency filter left
                no terms, if a second walk yields nothing, or if tf-idf weighs
                every term zero.
        """
        dictionary = corpora.Dictionary()
        counted = 0
        for batch in batched((p.lemmas for p in corpus()), _BATCH):
            dictionary.add_documents(batch, prune_at=None)  # pyright: ignore[reportArgumentType]
            counted += len(batch)

        if not counted:
            raise NoVocabulary(
                f"there are no {language} passages to fit a topic model over"
            )

        before = len(dictionary)
        dictionary.filter_extremes(
            no_below=self._no_below,
            no_above=self._no_above,
            keep_n=None,  # pyright: ignore[reportArgumentType]
        )
        if not dictionary:
            raise NoVocabulary(
                f"the frequency filter left no terms: {before} were found across "
                f"{counted} passage(s), and none appears in at least "
                f"{self._no_below} of them and at most {self._no_above:.0%}"
            )

        bows = _BagsOfWords(corpus, dictionary)
        # A corpus that cannot be walked twice.
        if not any(bool(bow) for bow in bows):
            raise NoVocabulary(
                f"a second walk over the corpus yielded none of the "
                f"{len(dictionary)} terms the first one found. A fit walks it "
                f"once per pass, so it must be re-readable."
            )

        # From the dictionary, which already holds the document frequencies.
        weighting = TfidfModel(dictionary=dictionary)
        # A term in every passage has an idf of zero. When every surviving term
        # does, the whole weighted matrix is zero, and the factorisation
        # divides by its norm and returns a model of NaN.
        if not any(bool(weighted) for weighted in weighting[bows]):
            raise NoVocabulary(
                f"every one of the {len(dictionary)} terms kept appears in all "
                f"{counted} {language} passage(s), so tf-idf weighs each of them "
                f"zero and nothing separates one passage from another. Lower "
                f"TOPIC_NO_ABOVE, which is {self._no_above:.0%}, to drop them."
            )

        log.info(
            "%s: fitting %d topics over %d passage(s), %d term(s) of %d kept",
            language,
            self._num_topics,
            counted,
            len(dictionary),
            before,
        )
        model = Nmf(
            corpus=weighting[bows],
            id2word=dictionary,
            num_topics=self._num_topics,
            passes=self._passes,
            random_state=self._random_state,
        )

        weights, without, space = self._score(model, corpus, dictionary, weighting)
        return Fitting(
            language=language,
            topics=self._topics(model),
            weights=weights,
            passages=counted,
            without_topics=without,
            vocabulary=len(dictionary),
            space=space,
        )

    def _topics(self, model: Nmf) -> list[FittedTopic]:
        """Reads each topic's signature out of the fitted model."""
        return [
            FittedTopic(
                topic_index=index,
                top_terms=[
                    term for term, _ in model.show_topic(index, topn=self._top_terms)
                ],
            )
            for index in range(model.num_topics)
        ]

    def _score(
        self,
        model: Nmf,
        corpus: Corpus,
        dictionary: corpora.Dictionary,
        weighting: TfidfModel,
    ) -> tuple[list[PassageWeight], int, TopicSpace]:
        """Scores every passage against the fitted model, in one walk.

        Returns:
            The memberships above the weight floor, how many passages hold
            none, and the space the visualisation is drawn from.
        """
        weights: list[PassageWeight] = []
        doc_topic: list[list[float]] = []
        doc_lengths: list[int] = []
        without = 0
        for passage in corpus():
            bow = cast("list[tuple[int, int]]", dictionary.doc2bow(passage.lemmas))
            found = model.get_document_topics(weighting[bow], minimum_probability=0)
            if found:
                row = [0.0] * model.num_topics
                for index, weight in found:
                    row[int(index)] = float(weight)
                total = sum(row)
                doc_topic.append([value / total for value in row])
                doc_lengths.append(sum(count for _, count in bow))

            kept = [
                (index, weight) for index, weight in found if weight > self._min_weight
            ]
            # Counts both a passage holding none of the vocabulary and one
            # whose every weight fell below the floor.
            if not kept:
                without += 1
                continue
            weights.extend(
                PassageWeight(
                    passage_id=passage.id,
                    topic_index=int(index),
                    # Clamped to the CHECK on passage_topics.weight.
                    weight=min(1.0, float(weight)),
                )
                for index, weight in kept
            )
        if without:
            log.warning(
                "%d passage(s) came out with no topic, so they are in no "
                "topic-weighted report",
                without,
            )
        return (
            weights,
            without,
            TopicSpace(
                topic_term=model.get_topics(normalize=True).tolist(),
                doc_topic=doc_topic,
                doc_lengths=doc_lengths,
                vocabulary=[dictionary[i] for i in range(len(dictionary))],
                term_frequency=[dictionary.cfs[i] for i in range(len(dictionary))],
            ),
        )


def carry_labels(
    fitted: list[FittedTopic], previous: list[FittedTopic]
) -> list[FittedTopic]:
    """Re-attaches a previous fit's labels to the topics that replaced them.

    Matched on shared top terms, above a Jaccard overlap of `_LABEL_MATCH`.
    Each previous label is used at most once, on its best match. The coverage
    flag travels with the label.

    Args:
        fitted: The topics this run produced.
        previous: The topics a person or an earlier run said something about.

    Returns:
        `fitted`, each topic carrying the label it matched or none.

    ponytail: a greedy pass, so a label can land on the second-best topic when
    two compete for it. The upgrade is to solve it as an assignment problem.
    """
    taken: set[int] = set()
    carried: list[FittedTopic] = []
    for topic in fitted:
        best, score = None, 0.0
        for candidate in previous:
            if candidate.topic_index in taken:
                continue
            overlap = _jaccard(topic.top_terms, candidate.top_terms)
            if overlap > score:
                best, score = candidate, overlap
        if best is None or score < _LABEL_MATCH:
            carried.append(topic)
            continue
        taken.add(best.topic_index)
        carried.append(
            FittedTopic(
                topic_index=topic.topic_index,
                top_terms=topic.top_terms,
                label=best.label,
                labelled_by=best.labelled_by,
                include_in_coverage=best.include_in_coverage,
            )
        )
    return carried


def _jaccard(left: list[str], right: list[str]) -> float:
    """Shared terms as a share of the terms either list holds, 0 for two empties."""
    first, second = set(left), set(right)
    union = first | second
    return len(first & second) / len(union) if union else 0.0
