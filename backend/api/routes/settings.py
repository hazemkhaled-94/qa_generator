"""Reading and changing what a service is configured to do.

One service per request: the six stages, and `platform` for the model, the
tokenizer and the language pipelines they share.

The shapes are here; what decides whether a change may be made is
`settings.changes`, shared with the command line.

Nothing here runs a stage. A setting reaches a worker when that worker next
claims a row, and what a change staled is named in the answer.
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

#: Which status code each refusal reads as.
_STATUS = {
    "unknown_setting": 404,
    "wrong_service": 400,
    "fixed_setting": 400,
    "not_a_choice": 400,
    "out_of_range": 400,
    "would_not_load": 400,
    "version_moved": 409,
}

#: The service a request names. A Literal, so the OpenAPI document lists
#: them and anything else is refused before it reaches the catalogue.
#:
#: Held identical to `settings.catalog.Service` by
#: `tests/static/test_api_vocabularies.py`. The two were written out twice
#: and drifted the first time a service was added: the catalogue grew
#: `assessment`, this did not, and `GET /settings/assessment` answered 422
#: for a service whose settings the catalogue was describing perfectly -
#: which reaches a person as a Configuration panel that will not open.
Service = Literal[
    "ingestion",
    "parsing",
    "chunking",
    "extraction",
    "topics",
    "questions",
    "assessment",
    "platform",
]

store = SettingsStore()


@dataclass(frozen=True)
class SettingState:
    """One setting, as a page needs it to draw a control.

    `value` is what the service reads now, `default` is what the files say,
    and the two differ exactly when `stored` is true. Both are null when the
    setting is absent.
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
    when it writes; a stale one is refused.
    """

    service: str
    version: str
    settings: list[SettingState]


class Change(BaseModel):
    """What a page is asking to change.

    `version` is optional. Absent, the write lands whatever has happened
    since the page was drawn.
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
    value. Nothing is requeued.
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
    setting named must belong to this service, and one refusal refuses the
    whole request.

    The next row a worker claims is worked under the new settings.

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
