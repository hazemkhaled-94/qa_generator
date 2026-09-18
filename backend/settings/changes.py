"""Deciding whether a change may be made, and making it.

Shared by the route and the command line. The HTTP shapes stay in the route
and the printing stays in the command line.

A value is written only if every service that could read it still parses its
settings afterwards, through the stage's own `Settings.load`. A change is
refused when it breaks something that was working, not when something was
already broken.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import extraction.config
import ingestion.config
import llm.config
import preprocessing.chunking.config
import preprocessing.parsing.config
import question_generation.config
import topic_modelling.config
from settings import catalog
from settings.store import Settings as Store

#: What must still parse after a change. All of them on every write, because
#: a platform setting is read by stages that were not named in the request.
#: `platform` is llm.config; the spaCy pipelines and the tokenizer are
#: checked by the stages that read them.
PARSERS: dict[str, Any] = {
    "ingestion": ingestion.config.Settings.load,
    "parsing": preprocessing.parsing.config.Settings.load,
    "chunking": preprocessing.chunking.config.Settings.load,
    "extraction": extraction.config.Settings.load,
    "topics": topic_modelling.config.Settings.load,
    "questions": question_generation.config.Settings.load,
    "platform": llm.config.Settings.load,
}


class Refused(Exception):
    """A change the deployment will not accept.

    Carries a stable code beside its sentence: the route answers with both,
    the command line prints the sentence.
    """

    def __init__(self, code: str, detail: str) -> None:
        """Initialises the refusal."""
        super().__init__(detail)
        self.code = code
        self.detail = detail


class Moved(Refused):
    """A change sent against a version that is no longer current."""

    def __init__(self, sent: str, current: str) -> None:
        """Says which version was expected and which one is there."""
        super().__init__(
            "version_moved",
            f"this was sent against version {sent} and the settings are at "
            f"{current}, so somebody has changed them since. Read them again "
            f"and decide against what is there now.",
        )


@dataclass(frozen=True)
class Outcome:
    """What one change moved, and what it left stale.

    `stale` names the stages whose stored output was produced under the old
    value. Nothing is requeued.
    """

    version: str
    changed: list[str] = field(default_factory=list)
    cleared: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def moved(self) -> list[str]:
        """Every setting this change wrote or returned to the files."""
        return [*self.changed, *self.cleared]


def owned(service: str, names: list[str]) -> dict[str, catalog.Setting]:
    """Looks up the settings a caller named, refusing what it may not write.

    Raises:
        Refused: `unknown_setting` for a name nothing configures,
            `wrong_service` for one another page owns, `fixed_setting` for
            one the deployment owns.
    """
    found = {}
    for name in names:
        try:
            setting = catalog.writable(name)
        except KeyError as exc:
            raise Refused("unknown_setting", str(exc)) from None
        except ValueError as exc:
            raise Refused("fixed_setting", str(exc)) from None
        if setting.service != service:
            raise Refused(
                "wrong_service",
                f"{name} is configured by {setting.service}, not by "
                f"{service}. One page configures one service.",
            )
        found[name] = setting
    return found


def offered(setting: catalog.Setting, value: str) -> list[str]:
    """What a value names, which is what `choices` is a list of.

    One value for `text`, every entry for a list, every key for a mix.
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


def within_bounds(setting: catalog.Setting, value: str) -> None:
    """Refuses a value the catalogue says is out of range or off the list.

    What the stage's own parser cannot say: a share above one parses as a
    number and is still a gate nothing passes.

    Raises:
        Refused: `not_a_choice` or `out_of_range`.
    """
    if setting.choices:
        unknown = [one for one in offered(setting, value) if one not in setting.choices]
        if unknown:
            raise Refused(
                "not_a_choice",
                f"{setting.name} takes {', '.join(setting.choices)}; "
                f"{', '.join(unknown)} is not among them.",
            )
    if setting.low is None and setting.high is None:
        return
    try:
        number = float(value)
    except ValueError:
        # Left to the stage's parser, which names the setting.
        return
    if setting.low is not None and number < setting.low:
        raise Refused(
            "out_of_range",
            f"{setting.name} may not be below {setting.low:g}; {value!r} is.",
        )
    if setting.high is not None and number > setting.high:
        raise Refused(
            "out_of_range",
            f"{setting.name} may not be above {setting.high:g}; {value!r} is.",
        )


def failing(source: Mapping[str, str]) -> dict[str, str]:
    """Which services fail to load from a source, and what each said."""
    refused = {}
    for service, load in PARSERS.items():
        try:
            load(source)
        except (KeyError, ValueError) as exc:
            # Stripped: a KeyError stringifies with its quotes.
            refused[service] = str(exc).strip("'\"")
    return refused


def still_parses(before: dict[str, str], candidate: Mapping[str, str]) -> None:
    """Refuses a change that stops a service that was working.

    Args:
        before: The services that do not load as things stand, by name.
        candidate: The source that would be read if the change landed.

    Raises:
        Refused: `would_not_load`, carrying the stage's own message.
    """
    for service, reason in failing(candidate).items():
        if service not in before:
            raise Refused(
                "would_not_load", f"{service} would not start with that: {reason}"
            )


def apply(
    service: str,
    values: Mapping[str, str | None],
    version: str | None = None,
    store: Store | None = None,
) -> Outcome:
    """Checks one change and writes it, or refuses the whole of it.

    Args:
        service: Which service is being configured.
        values: The settings to write, by name. A None returns one to
            whatever the files say.
        version: The version the caller decided against, or None to write
            whatever has happened since.
        store: The store to write through, built when not given.

    Returns:
        What moved, and what it left stale.

    Raises:
        Refused: If any one setting or value is refused. The whole change
            goes with it.
    """
    store = store or Store()
    resolved = store.resolved()
    at = store.version()
    if version is not None and version != at:
        raise Moved(version, at)

    settings = owned(service, list(values))
    for name, value in values.items():
        if value is not None:
            within_bounds(settings[name], value)

    # What the source would be if this landed. A null reads as the file.
    candidate = dict(resolved)
    for name, value in values.items():
        candidate[name] = os.environ.get(name, "") if value is None else value.strip()
    still_parses(failing(resolved), candidate)

    changed, cleared = [], []
    for name, value in values.items():
        if value is None:
            if store.clear(name):
                cleared.append(name)
        elif value.strip() != (resolved.get(name) or ""):
            store.write(name, value)
            changed.append(name)

    stale = sorted(
        {stage for name in changed + cleared for stage in settings[name].invalidates}
    )
    return Outcome(
        version=store.version(),
        changed=changed,
        cleared=cleared,
        stale=stale,
        detail=_detail(changed + cleared, stale),
    )


def _detail(moved: list[str], stale: list[str]) -> str:
    """Says what happened, and what is now worth rebuilding."""
    if not moved:
        return "nothing changed; every value given is the one already set."
    written = (
        f"{len(moved)} setting(s) changed. Workers pick them up on the row "
        f"they claim next."
    )
    if not stale:
        return written
    return (
        f"{written} What {', '.join(stale)} already produced was made under "
        f"the old values; rerun that stage to rebuild it."
    )
