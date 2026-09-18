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
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

import extraction.config
import ingestion.config
import llm.config
import preprocessing.chunking.config
import preprocessing.parsing.config
import question_generation.config
import topic_modelling.config
from api.errors import ApiError, ErrorBody
from settings import catalog
from settings.store import Settings as SettingsStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

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

#: What must still parse after a change. Every one of them on every write,
#: not just the service being changed: EMBEDDING_MODEL belongs to the
#: platform and is read by chunking and by question generation, so a change
#: to it that only the platform had to accept would be a change that stops
#: two workers.
#:
#: `platform` is llm.config, which is the only settings object the shared
#: values make on their own. The spaCy pipelines and the tokenizer are read
#: where they are used rather than loaded into a dataclass, so what checks
#: those is the stage that reads them.
_PARSERS: dict[str, Any] = {
    "ingestion": ingestion.config.Settings.load,
    "parsing": preprocessing.parsing.config.Settings.load,
    "chunking": preprocessing.chunking.config.Settings.load,
    "extraction": extraction.config.Settings.load,
    "topics": topic_modelling.config.Settings.load,
    "questions": question_generation.config.Settings.load,
    "platform": llm.config.Settings.load,
}

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


def _owned(service: str, names: list[str]) -> list[catalog.Setting]:
    """Looks up the settings a request named, refusing what it may not write.

    Raises:
        ApiError: 404 `unknown_setting` for a name nothing configures, 400
            `wrong_service` for one another page owns, 400 `fixed_setting`
            for one the deployment owns.
    """
    found = []
    for name in names:
        try:
            setting = catalog.writable(name)
        except KeyError as exc:
            raise ApiError(404, "unknown_setting", str(exc)) from None
        except ValueError as exc:
            raise ApiError(400, "fixed_setting", str(exc)) from None
        if setting.service != service:
            raise ApiError(
                400,
                "wrong_service",
                f"{name} is configured by {setting.service}, not by "
                f"{service}. One page configures one service.",
            )
        found.append(setting)
    return found


def _offered(setting: catalog.Setting, value: str) -> list[str]:
    """What a value names, which is what `choices` is a list of.

    A closed set constrains one value for a `text` setting, every entry for
    a list, and every key for a mix. Checking the whole string against the
    list instead would make every setting that takes more than one thing
    unwritable: `EXTRACTION_KINDS=summary,outline` names two kinds that are
    both on the list and is not itself on it.
    """
    if setting.kind == "csv":
        return [part.strip() for part in value.split(",") if part.strip()]
    if setting.kind == "mapping":
        return [
            entry.split(":", 1)[0].strip()
            for entry in value.split(",")
            if entry.strip()
        ]
    return [value]


def _within_bounds(setting: catalog.Setting, value: str) -> None:
    """Refuses a value the catalogue says is out of range or not on the list.

    What the stage's own parser cannot say. `QUESTIONS_ANSWER_OVERLAP=2` is
    a number and parses; it is also a share of something, and a share above
    one is a gate nothing can pass.

    Raises:
        ApiError: 400 `out_of_range` or `not_a_choice`.
    """
    if setting.choices:
        unknown = [
            one for one in _offered(setting, value) if one not in setting.choices
        ]
        if unknown:
            raise ApiError(
                400,
                "not_a_choice",
                f"{setting.name} takes {', '.join(setting.choices)}; "
                f"{', '.join(unknown)} is not among them.",
            )
    if setting.low is None and setting.high is None:
        return
    try:
        number = float(value)
    except ValueError:
        # Left to the stage's parser, which names the setting and says what
        # it wanted. Refusing here would say it twice, differently.
        return
    if setting.low is not None and number < setting.low:
        raise ApiError(
            400,
            "out_of_range",
            f"{setting.name} may not be below {setting.low:g}; {value!r} is.",
        )
    if setting.high is not None and number > setting.high:
        raise ApiError(
            400,
            "out_of_range",
            f"{setting.name} may not be above {setting.high:g}; {value!r} is.",
        )


def _parses(source: dict[str, str]) -> dict[str, str]:
    """Which services fail to load from a source, and what each said."""
    failed = {}
    for service, load in _PARSERS.items():
        try:
            load(source)
        except (KeyError, ValueError) as exc:
            # KeyError stringifies with its quotes, which reads badly in a
            # message a person is shown.
            failed[service] = str(exc).strip("'\"")
    return failed


def _still_parses(failing: dict[str, str], candidate: dict[str, str]) -> None:
    """Refuses a change that stops a service that was working.

    Compared against what the deployment does now rather than judged on its
    own, so a service already failing for a reason nobody is changing does
    not block an unrelated setting. A deployment with no model configured
    can still change a parsing threshold.

    Args:
        failing: The services that do not load as things stand, by name.
        candidate: The source that would be read if the change landed.

    Raises:
        ApiError: 400 `would_not_load`, carrying the stage's own message.
    """
    for service, reason in _parses(candidate).items():
        if service not in failing:
            raise ApiError(
                400,
                "would_not_load",
                f"{service} would not start with that: {reason}",
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
    described, resolved = _described(service)
    if change.version is not None and change.version != described.version:
        raise ApiError(
            409,
            "version_moved",
            f"this was sent against version {change.version} and the "
            f"settings are at {described.version}, so somebody has changed "
            f"them since the page was drawn. Read them again and decide "
            f"against what is there now.",
        )

    settings = {one.name: one for one in _owned(service, list(change.values))}
    for name, value in change.values.items():
        if value is not None:
            _within_bounds(settings[name], value)

    # What the source would be if this landed. A null is what the files say,
    # which is what clearing the override leaves behind.
    candidate = dict(resolved)
    for name, value in change.values.items():
        candidate[name] = os.environ.get(name, "") if value is None else value.strip()
    _still_parses(_parses(resolved), candidate)

    changed, cleared = [], []
    for name, value in change.values.items():
        if value is None:
            if store.clear(name):
                cleared.append(name)
        elif value.strip() != (resolved.get(name) or ""):
            store.write(name, value)
            changed.append(name)

    moved = changed + cleared
    stale = sorted({stage for name in moved for stage in settings[name].invalidates})
    return Changed(
        service=service,
        version=store.version(),
        changed=changed,
        cleared=cleared,
        stale=stale,
        detail=_detail(moved, stale),
    )


def _detail(moved: list[str], stale: list[str]) -> str:
    """Says what happened, and what is now worth rebuilding."""
    if not moved:
        return "nothing changed; every value given is the one already set."
    written = f"{len(moved)} setting(s) changed. Workers pick them up on the row they claim next."
    if not stale:
        return written
    return (
        f"{written} What {', '.join(stale)} already produced was made under "
        f"the old values; rerun that stage to rebuild it."
    )
