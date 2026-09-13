"""Naming a topic, and carrying a name across a refit.

A refit deletes every topic, so a hand-typed label survives only by being
matched on top terms.
"""

from __future__ import annotations

from topic_modelling.labels import TopicLabeller
from topic_modelling.models import FittedTopic
from topic_modelling.topics import carry_labels

FIRST = ["lieferung", "versand", "transport", "paket", "monatlich"]
SECOND = ["wartung", "reparatur", "instandhaltung", "ersatzteil", "pruefung"]


class _Answer:
    """What the client hands back."""

    def __init__(self, label: str) -> None:
        """Initialises the answer with its label."""
        self.label = label


class _Model:
    """A client that answers whatever it was told to, and counts calls."""

    def __init__(self, says: str) -> None:
        """Initialises the client with the answer it gives."""
        self.says, self.calls = says, 0
        self.model = "test-model"

    def answer(self, **_: object) -> _Answer:
        """Answers, and records that it was asked."""
        self.calls += 1
        return _Answer(self.says)


def test_a_model_answer_becomes_the_label() -> None:
    """The label is taken as given, and the model is recorded."""
    labeller = TopicLabeller(_Model('"Anti-money laundering".'), {"en": "English"})

    named = labeller.label(FittedTopic(0, ["money", "laundering"]), "en", ["excerpt"])
    assert named == "Anti-money laundering", named
    assert labeller.model == "test-model"


def test_a_refusal_is_not_stored_as_a_name() -> None:
    """A model that finds no subject leaves the topic unnamed."""
    refused = TopicLabeller(_Model("Mixed"), {"en": "English"})

    assert refused.label(FittedTopic(0, ["a", "b"]), "en", []) is None


def test_a_topic_a_person_named_is_never_offered_to_the_model() -> None:
    """The person's label and its provenance both stay."""
    from topic_modelling.service import TopicModellingService

    counting = _Model("Generated")
    service = TopicModellingService.__new__(TopicModellingService)
    service._labeller = TopicLabeller(counting, {"en": "English"})
    service._repository = type("R", (), {"excerpts": staticmethod(lambda ids: [])})()
    fitting = type("F", (), {"weights": []})()

    named = service._named(
        [
            FittedTopic(0, ["a"], label="Typed by a person", labelled_by="person"),
            FittedTopic(1, ["b"]),
        ],
        fitting,
        "en",
    )

    assert named[0].label == "Typed by a person", named[0]
    assert named[0].labelled_by == "person", named[0]
    assert named[1].label == "Generated", named[1]
    assert named[1].labelled_by == "test-model", named[1]
    assert counting.calls == 1, f"the model was asked {counting.calls} times, not 1"


def test_a_label_travels_to_the_topic_it_matches() -> None:
    """The top terms decide, not the topic number."""
    previous = [
        FittedTopic(0, FIRST, label="Shipping", include_in_coverage=False),
        FittedTopic(1, SECOND, label="Maintenance"),
    ]
    refitted = [
        FittedTopic(
            0, ["wartung", "reparatur", "instandhaltung", "ersatzteil", "intervall"]
        ),
        FittedTopic(1, ["lieferung", "versand", "transport", "paket", "jaehrlich"]),
    ]

    carried = carry_labels(refitted, previous)
    assert carried[0].label == "Maintenance", carried[0].label
    assert carried[1].label == "Shipping", carried[1].label
    assert carried[1].include_in_coverage is False, "the exclusion must travel too"


def test_a_label_is_dropped_rather_than_put_on_an_unrelated_topic() -> None:
    """No match means no name."""
    previous = [
        FittedTopic(0, FIRST, label="Shipping", include_in_coverage=False),
        FittedTopic(1, SECOND, label="Maintenance"),
    ]

    unrelated = carry_labels(
        [FittedTopic(0, ["farbe", "lack", "oberflaeche"])], previous
    )
    assert unrelated[0].label is None, unrelated[0].label


def test_a_label_is_used_once() -> None:
    """Two topics matching the same label take it between them."""
    previous = [
        FittedTopic(0, FIRST, label="Shipping", include_in_coverage=False),
        FittedTopic(1, SECOND, label="Maintenance"),
    ]

    twice = carry_labels([FittedTopic(0, FIRST), FittedTopic(1, FIRST)], previous)
    assert [t.label for t in twice].count("Shipping") == 1, [t.label for t in twice]
