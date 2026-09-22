"""Pushing rows out for review, and pulling the verdicts back.

Two directions and one rule: the database decides. Argilla holds a copy of
what was put in front of somebody and the answers they gave, and `pull`
brings the answers home. Delete the Argilla dataset and nothing in the
pipeline is lost; delete the rows and the copy is meaningless.

A pull takes only submitted answers. Argilla keeps a draft the moment a
reviewer touches a record, and a draft is somebody part-way through
thinking rather than a decision.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import argilla as rg

from review import datasets
from review.config import Settings
from review.records import ReviewRepository

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Catalogs:
    """Where each kind of verdict is read from and written back to.

    The existing catalogues, not new queries. A topic renamed here and a
    topic renamed on the Topics page have to be the same write, or the
    provenance the two record drifts apart - `describe` is what sets
    labelled_by to `person`, and a second path that forgot to would leave
    a human label looking like the model's.
    """

    #: Facts: the sample, and the only verdict with no route of its own.
    facts: ReviewRepository
    #: Questions: `decide`, which PATCH /questions/{id} also calls.
    questions: Any
    #: Topics: `describe`, which PATCH /topics/{id} also calls.
    topics: Any


def connect(settings: Settings) -> rg.Argilla:
    """Opens the client and makes sure the workspace is there."""
    client = rg.Argilla(api_url=settings.api_url, api_key=settings.api_key)
    if client.workspaces(settings.workspace) is None:
        log.info("creating workspace %s", settings.workspace)
        client.workspaces.add(rg.Workspace(name=settings.workspace))
    return client


def _dataset(client: rg.Argilla, settings: Settings, name: str) -> rg.Dataset:
    """The dataset by that name, created on first push.

    Not recreated if it is there: a dataset holds the answers given so far,
    and a push that dropped it would throw away every verdict not yet
    pulled.
    """
    found = client.datasets(name=name, workspace=settings.workspace)
    if found is not None:
        return found
    log.info("creating dataset %s in %s", name, settings.workspace)
    return rg.Dataset(
        name=name,
        workspace=settings.workspace,
        settings=datasets.SETTINGS[name](),
        client=client,
    ).create()


def push(
    name: str,
    settings: Settings,
    catalogs: Catalogs,
    ids: Sequence[int] | None = None,
) -> int:
    """Puts a sample of one kind of row in front of a reviewer.

    `ids` names exactly which questions to push instead of sampling, which
    is how a queue built elsewhere - `evaluation.second_opinion`'s
    disagreements - reaches a reviewer. Questions only; the other two sets
    have no such queue behind them.

    Returns:
        How many records were written.
    """
    client = connect(settings)
    dataset = _dataset(client, settings, name)

    if name == datasets.FACTS:
        records = [
            datasets.fact_record(one) for one in catalogs.facts.facts(settings.sample)
        ]
    elif name == datasets.QUESTIONS:
        records = [
            datasets.question_record(one)
            for one in catalogs.facts.questions(settings.sample, ids)
        ]
    else:
        # Every topic, not a sample. A corpus has dozens, not thousands,
        # and a topic left unreviewed is one whose name nobody checked.
        records = [datasets.topic_record(one) for one in catalogs.topics.topics()]

    if not records:
        log.info("%s: nothing left to review", name)
        return 0
    dataset.records.log(records)
    log.info("%s: %d record(s) pushed to %s", name, len(records), settings.api_url)
    return len(records)


def pull(name: str, settings: Settings, catalogs: Catalogs) -> int:
    """Brings submitted verdicts back into the database.

    Returns:
        How many rows were written.
    """
    client = connect(settings)
    dataset = client.datasets(name=name, workspace=settings.workspace)
    if dataset is None:
        log.info("%s: no such dataset; nothing has been pushed yet", name)
        return 0

    answered = list(dataset.records(with_responses=True))

    if name == datasets.FACTS:
        written = catalogs.facts.review_facts(
            [
                (int(record.metadata["fact_id"]), verdict)
                for record in answered
                if (verdict := _answer(record, "verdict")) is not None
            ]
        )
    elif name == datasets.QUESTIONS:
        written = sum(
            bool(
                catalogs.questions.decide(int(record.metadata["question_id"]), verdict)
            )
            for record in answered
            if (verdict := _answer(record, "verdict")) is not None
        )
    else:
        written = sum(
            bool(
                catalogs.topics.describe(
                    int(record.metadata["topic_id"]),
                    label=_answer(record, "corrected_label"),
                    # Absent means unanswered, and an unanswered coverage
                    # question must not read as "take it out of coverage".
                    include_in_coverage=_answer(record, "in_coverage") != "no",
                )
            )
            for record in answered
            if _answer(record, "corrected_label") is not None
            or _answer(record, "in_coverage") is not None
        )

    log.info("%s: %d verdict(s) written", name, written)
    return written


def _answer(record, question: str) -> str | None:
    """What a reviewer submitted for one question, or None.

    Drafts are skipped. Argilla saves one the moment a record is touched,
    so a draft is somebody part-way through rather than a decision, and
    writing it back would record an opinion nobody has finished having.
    """
    for response in record.responses:
        if response.question_name == question and str(response.status) in (
            "submitted",
            "ResponseStatus.submitted",
        ):
            return response.value
    return None
