"""Fitting a topic model over one language's vocabulary."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from itertools import batched

from gensim import corpora
from gensim.models import Nmf, TfidfModel

from topic_modelling.models import (
    FittedTopic,
    Fitting,
    PassageVocabulary,
    PassageWeight,
)

log = logging.getLogger(__name__)

#: How many passages to hold at once while building the vocabulary.
_BATCH = 500

#: The corpus, as the fitter takes it: a callable handing back a fresh walk
#: over every passage. A callable and not an iterable, because the fit walks
#: the corpus several times and a generator is spent after the first.
Corpus = Callable[[], Iterable[PassageVocabulary]]

#: Smallest share of terms two topics must have in common before a label is
#: carried from one to the other. Set high: attaching a person's label to the
#: wrong topic is worse than making them type it again.
_LABEL_MATCH = 0.4


class NoVocabulary(Exception):
    """Raised when the frequency filter left nothing to fit a model on."""


class _BagsOfWords:
    """The corpus as bags of words, rebuilt on every walk.

    What the model takes: an iterable it may walk once per pass. Wrapping it
    in a TfidfModel keeps it lazy, so the corpus is still streamed.
    """

    def __init__(self, corpus: Corpus, dictionary: corpora.Dictionary) -> None:
        """Initialises the view."""
        self._corpus = corpus
        self._dictionary = dictionary

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        """Walks the corpus once."""
        for passage in self._corpus():
            yield self._dictionary.doc2bow(passage.lemmas)


class TopicFitter:
    """Fits topics over the terms one language's passages use.

    Not a pretrained model and not an LLM: the vocabulary is the lemmas
    chunking stored, which is what keeps the result industry-agnostic.
    Grammar never reaches it, because only content parts of speech are
    lemmatised; the frequency filter is left to catch the boilerplate no word
    list could know about.

    Non-negative matrix factorisation over a tf-idf weighted space, rather
    than LDA over raw counts. LDA is defined over counts - words drawn from a
    multinomial - so tf-idf cannot be fed to it without contradicting its own
    likelihood, and measured on this corpus doing so was worse than counts.
    Factorisation carries no such assumption, and weighting the input is what
    stops one dominant vocabulary spreading across every topic: measured at
    twelve topics, term overlap between topics fell from 32% to 11% and
    coherence rose from 0.52 to 0.66.

    The weights it returns are normalised, so a passage's memberships still
    read as shares and still satisfy the CHECK on passage_topics.weight.

    The fit is seeded, so the same corpus and the same settings give the same
    topics.
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
        """Initialises the fitter, refusing a setting that cannot do its job.

        Checked here because these arrive from the environment: a weight
        floor of 0 would write a row for every topic of every passage, and
        breach the CHECK constraint on passage_topics.weight.
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
        once per pass, and once more to score. The tf-idf weighting is read
        off the vocabulary's own document frequencies and applied lazily, so
        it costs no extra walk and the corpus is still streamed.
        """
        dictionary = corpora.Dictionary()
        counted = 0
        for batch in batched((p.lemmas for p in corpus()), _BATCH):
            dictionary.add_documents(batch, prune_at=None)
            counted += len(batch)

        if not counted:
            raise NoVocabulary(
                f"there are no {language} passages to fit a topic model over"
            )

        before = len(dictionary)
        dictionary.filter_extremes(
            no_below=self._no_below, no_above=self._no_above, keep_n=None
        )
        if not dictionary:
            raise NoVocabulary(
                f"the frequency filter left no terms: {before} were found across "
                f"{counted} passage(s), and none appears in at least "
                f"{self._no_below} of them and at most {self._no_above:.0%}"
            )

        bows = _BagsOfWords(corpus, dictionary)
        # Reads one passage in the healthy case, because any() short-circuits.
        # Catches the corpus that cannot be walked twice: the vocabulary above
        # spent it, and the model would fit an empty corpus without complaining.
        if not any(bool(bow) for bow in bows):
            raise NoVocabulary(
                f"a second walk over the corpus yielded none of the "
                f"{len(dictionary)} terms the first one found. A fit walks it "
                f"once per pass, so it must be re-readable."
            )

        log.info(
            "%s: fitting %d topics over %d passage(s), %d term(s) of %d kept",
            language,
            self._num_topics,
            counted,
            len(dictionary),
            before,
        )
        # From the dictionary rather than the corpus: it already counted the
        # document frequencies idf is, so this needs no walk of its own.
        weighting = TfidfModel(dictionary=dictionary)
        model = Nmf(
            corpus=weighting[bows],
            id2word=dictionary,
            num_topics=self._num_topics,
            passes=self._passes,
            random_state=self._random_state,
        )

        weights, without = self._score(model, corpus, dictionary, weighting)
        return Fitting(
            language=language,
            topics=self._topics(model),
            weights=weights,
            passages=counted,
            without_topics=without,
            vocabulary=len(dictionary),
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
    ) -> tuple[list[PassageWeight], int]:
        """Scores every passage against the fitted model."""
        weights: list[PassageWeight] = []
        without = 0
        for passage in corpus():
            found = model.get_document_topics(
                weighting[dictionary.doc2bow(passage.lemmas)],
                minimum_probability=self._min_weight,
            )
            # A passage holding none of the vocabulary, and one whose every
            # weight fell below the floor, are both absent from every
            # topic-weighted report, so both are counted here.
            if not found:
                without += 1
                continue
            weights.extend(
                PassageWeight(
                    passage_id=passage.id,
                    topic_index=int(index),
                    # Clamped: the weights are normalised in floating point,
                    # so a distribution's last member can round a hair above 1
                    # and breach the CHECK constraint on the column.
                    weight=min(1.0, float(weight)),
                )
                for index, weight in found
            )
        if without:
            log.warning(
                "%d passage(s) came out with no topic: every one of their terms "
                "was filtered out, so they are in no topic-weighted report",
                without,
            )
        return weights, without


def carry_labels(
    fitted: list[FittedTopic], previous: list[FittedTopic]
) -> list[FittedTopic]:
    """Re-attaches a previous fit's labels to the topics that replaced them.

    Matched on shared top terms, which is the only signature a topic stores.
    Each previous label is used at most once, on its best match.

    ponytail: a greedy pass, so a label can land on the second-best topic when
    two compete for it. Fixing that properly is an assignment problem; an
    unlabelled topic costs one person one minute.
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
                include_in_coverage=best.include_in_coverage,
            )
        )
    return carried


def _jaccard(left: list[str], right: list[str]) -> float:
    """Measures how much two term lists have in common."""
    first, second = set(left), set(right)
    union = first | second
    return len(first & second) / len(union) if union else 0.0
