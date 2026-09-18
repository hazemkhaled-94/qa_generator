"""Reading and changing what a service is configured to do.

One service per request, because one page configures one service. The stages
are `ingestion`, `parsing`, `chunking`, `extraction`, `topics` and
`questions`; `platform` is the model, the tokenizer and the language
pipelines the six of them share.

A value is written only if every service that could read it still parses its
settings afterwards. The parser is the stage's own `Settings.load`, handed
the environment overlaid with what is stored and with what this request
proposes - so what a page is refused with is the message the worker would
have failed at start-up with, written once, in the stage.

A change is refused when it breaks something that worked, and not when
something was already broken. A deployment that has never configured a model
should still be able to change a parsing threshold, and would not be if any
failure at all were enough to refuse.

Nothing here runs a stage. A setting reaches a worker when that worker next
claims a row; what a change staled is named in the answer, and rebuilding it
is the stage's own rerun.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.errors import ApiError, ErrorBody
from settings import catalog, changes
from settings.store import Settings as SettingsStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

#: What each refusal is answered with. The decision is `settings.changes`,
#: shared with the command line; which status code it reads as is HTTP's and
#: belongs here.
_STATUS = {
    "unknown_setting": 404,
    "wrong_service": 400,
    "fixed_setting": 400,
    "not_a_choice": 400,
    "out_of_range": 400,
    "would_not_load": 400,
    "version_moved": 409,
}

#: The service a request names. A Literal rather than a free string, so the
#: OpenAPI document lists them and anything else is refused before it reaches
#: the catalogue.
Service = Literal[
    "ingestion",
    "parsing",
    "chunking",
    "extraction",
    "topics",
    "questions",
    "platform",
]

store = SettingsStore()


@dataclass(frozen=True)
class SettingState:
    """One setting, as a page needs it to draw a control.

    `value` is what the service reads now and `default` is what the files
    say. The two differ exactly when `stored` is true, which is what a page
    marks and what its reset button clears.

    Both are null when a setting is absent, which is a state only a setting
    whose absence means something can be in.
    """

    name: str
    kind: str
    help: str
    value: str | None
    default: str | None
    stored: bool
    optional: bool
    fixed: bool
    invalidates: list[str]
    changed_at: datetime | None = None
    low: float | None = None
    high: float | None = None
    choices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ServiceSettings:
    """Everything one service is configured by.

    `version` names this configuration by its content. A page hands it back
    when it writes, and a write carrying a version that is no longer current
    is refused rather than landing on a change nobody saw.
    """

    service: str
    version: str
    settings: list[SettingState]


class Change(BaseModel):
    """What a page is asking to change.

    `version` is optional, and giving it is what makes a write safe against
    a second person changing the same service. Absent, the write lands
    whatever has happened since the page was drawn.
    """

    values: dict[str, str | None] = Field(
        description="The settings to write, by name. Null returns one to "
        "whatever the files say, which is the same as deleting the override."
    )
    version: str | None = Field(
        default=None,
        description="The version the page was drawn from. Refused with 409 "
        "if the configuration has changed since.",
    )


@dataclass(frozen=True)
class Changed:
    """What one write moved, and what it left stale.

    `stale` names the stages whose stored output was produced under the old
    value. Nothing is requeued here: which of them to rebuild, and when, is
    a decision with a corpus-sized cost behind it.
    """

    service: str
    version: str
    changed: list[str]
    cleared: list[str]
    stale: list[str]
    detail: str


def _described(service: str) -> tuple[ServiceSettings, dict[str, str]]:
    """Reads one service's settings, and the source they resolved from."""
    resolved = store.resolved()
    overrides = store.overrides()
    changed_at = store.changed_at()

    return (
        ServiceSettings(
            service=service,
            version=store.version(),
            settings=[
                SettingState(
                    name=one.name,
                    kind=one.kind,
                    help=one.help,
                    value=resolved.get(one.name) or None,
                    default=os.environ.get(one.name) or None,
                    stored=one.name in overrides,
                    optional=one.optional,
                    fixed=one.fixed,
                    invalidates=list(one.invalidates),
                    changed_at=changed_at.get(one.name),
                    low=one.low,
                    high=one.high,
                    choices=list(one.choices),
                )
                for one in catalog.of(service)
            ],
        ),
        resolved,
    )


@router.get("/{service}")
def read(service: Service) -> ServiceSettings:
    """Reports what one service is configured to do.

    Every setting the service reads, what it resolves to now, what the files
    say it would be, and whether somebody has changed it. A page draws its
    controls from this and holds no list of its own.
    """
    described, _ = _described(service)
    return described


@router.patch(
    "/{service}",
    responses={code: {"model": ErrorBody} for code in (400, 404, 409)},
)
def write(service: Service, change: Change) -> Changed:
    """Changes what one service is configured to do.

    A value of null returns that setting to whatever the files say. Every
    setting named must belong to this service, and the whole request is
    refused if any one of them is refused: half a change is a configuration
    nobody asked for.

    The next row a worker claims is worked under the new settings. Nothing
    is requeued, and what was produced under the old ones is named in
    `stale`.

    Raises:
        ApiError: 404 `unknown_setting`; 400 `wrong_service`,
            `fixed_setting`, `not_a_choice`, `out_of_range` or
            `would_not_load`; 409 `version_moved`.
    """
    try:
        moved = changes.apply(service, change.values, change.version, store)
    except changes.Refused as refusal:
        raise ApiError(_STATUS[refusal.code], refusal.code, refusal.detail) from None

    return Changed(
        service=service,
        version=moved.version,
        changed=moved.changed,
        cleared=moved.cleared,
        stale=moved.stale,
        detail=moved.detail,
    )
