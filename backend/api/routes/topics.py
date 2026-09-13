"""Routes for the topic modelling stage.

Reads and writes the queue and never works it, like every other stage. Asking
is what creates the row, so /discover replaces /start and there is no /rerun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from api.dependencies import export_bucket, topic_catalog, topics_queue
from api.errors import ApiError, ErrorBody
from api.routes.stage import StageAction, StageStatus, answered, status_of
from topic_modelling.models import StoredTopic, TopicFit, TopicRemoval

router = APIRouter(prefix="/topics", tags=["topics"])

#: An ISO 639-1 code, as the passages table records one.
_LANGUAGE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True)
class TopicDiscovery:
    """The answer to a request to fit the topics again."""

    fit: int
    detail: str


class TopicDescription(BaseModel):
    """What a person decided about one topic.

    Both survive a refit by being matched on top terms, which is why they are
    worth recording at all.
    """

    label: str | None = Field(default=None, max_length=120)
    include_in_coverage: bool = True


@router.get("/status")
def status() -> StageStatus:
    """Reports how many fits are queued and whether a worker is on one.

    `modelled` is the number of topics; the other states describe a fit.
    """
    return status_of("topics", topics_queue)


@router.get("")
def topics() -> list[StoredTopic]:
    """Lists the fitted topics with how much of the corpus each holds."""
    return topic_catalog.topics()


@router.patch("/{topic_id}", responses={404: {"model": ErrorBody}})
def describe(topic_id: int, description: TopicDescription) -> StoredTopic:
    """Names a topic, or takes it out of coverage reporting."""
    described = topic_catalog.describe(
        topic_id,
        label=description.label,
        include_in_coverage=description.include_in_coverage,
    )
    if described is None:
        raise ApiError(404, "unknown_topic", "no topic with that id")
    return described


@router.get("/fit")
def fit() -> TopicFit:
    """Reports the state of the topic model as a whole.

    Carries the live passage and membership counts beside the ones the fit
    recorded, so a model left stale by a re-chunk is visible rather than
    reading as healthy.
    """
    return topic_catalog.fit_state()


@router.get(
    "/visualisation/{language}",
    responses={code: {"model": ErrorBody} for code in (400, 404)},
    response_class=Response,
)
def visualisation(language: str) -> Response:
    """Serves one language's topic model as a pyLDAvis page.

    Produced by a fit, so a language modelled before this route existed has
    none until the next one.

    Raises:
        ApiError: 400 `invalid_language` if it is not an ISO 639-1 code, 404
            `no_visualisation` if no fit has drawn that language.
    """
    if not _LANGUAGE.fullmatch(language):
        raise ApiError(400, "invalid_language", "not an ISO 639-1 language code")

    drawn = export_bucket.find(export_bucket.topic_visualisation_key(language))
    if drawn is None:
        raise ApiError(
            404,
            "no_visualisation",
            f"no topic visualisation for {language}; fit the model to draw one",
        )
    return Response(content=drawn, media_type=export_bucket.TOPIC_VISUALISATION_TYPE)


@router.post("/discover", status_code=202)
def discover() -> TopicDiscovery:
    """Queues a fit over the whole corpus.

    Returns at once. Every existing topic and membership is replaced when the
    fit succeeds, and left alone when it does not. Asking twice queues one
    fit.
    """
    fit_id = topics_queue.request()
    return TopicDiscovery(
        fit=fit_id,
        detail=(
            f"fit {fit_id} queued; the worker will pick it up and replace every "
            "existing topic."
        ),
    )


@router.post("/stop")
def stop() -> StageAction:
    """Takes a queued fit back off the queue."""
    return answered("topics", "stop", topics_queue.stop())


@router.post("/retry")
def retry() -> StageAction:
    """Returns a failed fit to the queue."""
    return answered("topics", "retry", topics_queue.retry())


@router.delete("")
def delete() -> TopicRemoval:
    """Deletes every topic, and with it every membership and visualisation.

    Passages, facts and questions stay. Labels a person assigned go with the
    topics. Any queued fit goes too.
    """
    removed = topic_catalog.delete_all()
    for language in removed.languages:
        export_bucket.remove(export_bucket.topic_visualisation_key(language))
    return removed
