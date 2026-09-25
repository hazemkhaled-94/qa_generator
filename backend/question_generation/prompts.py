"""Every prompt this stage sends, for the record that resolves a version.

`questions.prompt_version` names a version; this is what says what that
version asked for. Composed here rather than stored as pieces, because a
type's system prompt is the shared rules plus its own directive and worked
examples assembled at call time, and the thing worth keeping is what the
model was actually given.

Three versions, not one. The writer's prompts, the phrasing judgements and
the verifier's four are bumped separately because they change for different
reasons: what a question IS, what counts as naming its source, and what
counts as recovering an answer.

Read by `run.py` at start-up and by the api, which serves it to the
Questions page. Nothing here calls a model or touches a row.
"""

from __future__ import annotations

from collections.abc import Iterator

from question_generation import phrasing, verifier
from question_generation.generation import (
    _FOLLOW,
    _PERTURB,
    _USER,
    _USER_FOLLOW,
    _Answered,
    _Unanswered,
)
from question_generation.types import PROMPT_VERSION, SPECS
from stages.prompts import Composed

#: What the stage records its prompts under.
SERVICE = "questions"

#: The JSON schemas the three writer shapes and the four judge shapes ask
#: their answers to come back in.
_ANSWERED = _Answered.model_json_schema()
_UNANSWERED = _Unanswered.model_json_schema()


def catalogue() -> list[Composed]:
    """Every prompt this stage can send, composed as it would be sent.

    Both halves and the shape: the system prompt, the user message as its
    template, and the JSON schema the answer has to come back in.
    """
    return list(_writer()) + list(_judges())


def _writer() -> Iterator[Composed]:
    """The prompts that write a question, at `types.PROMPT_VERSION`.

    Each type twice. `spans` adds the instruction to draw on more than one
    passage, and a type that does not span by itself is still sent the
    spanning prompt when its sample happens to cross passages - so the two
    are different prompts and a row for only one of them would be a record
    of half of what was sent.
    """
    for name, spec in sorted(SPECS.items()):
        yield Composed(name, PROMPT_VERSION, spec.system(), _USER, _ANSWERED)
        spanning = spec.system(spans=True)
        if spanning != spec.system():
            yield Composed(
                f"{name} (spans)", PROMPT_VERSION, spanning, _USER, _ANSWERED
            )
    yield Composed("perturb", PROMPT_VERSION, _PERTURB, _USER, _UNANSWERED)
    yield Composed("follow", PROMPT_VERSION, _FOLLOW, _USER_FOLLOW, _ANSWERED)


def _judges() -> Iterator[Composed]:
    """The prompts that judge one, each at its own module's version.

    The three phrasing judgements are a model's opinion about wording; the
    verifier's four are the round trip. `_VERIFY` asks for the answer,
    and the three below it ask whether passages support an answer already
    in hand - which is a different question, and the reason recoverability
    may cost a second call.

    All THREE phrasing prompts. `_NAMES` was missing, and a missing one is
    the one failure this catalogue has: the stage sends it - `_phrasing`
    asks it wherever the parse has already called a question thin - the row
    records the version, and only the lookup that resolves the version to a
    text comes back without it.
    """
    for name, text, user, shape in (
        (
            "phrasing: source",
            phrasing._SOURCE,
            phrasing._USER,
            phrasing._NamesItsSource,
        ),
        (
            "phrasing: names something",
            phrasing._NAMES,
            phrasing._USER,
            phrasing._NamesSomething,
        ),
        (
            "phrasing: self-contained",
            phrasing._CONTAINED,
            phrasing._USER_CONTAINED,
            phrasing._SelfContained,
        ),
    ):
        yield Composed(
            name, phrasing.PROMPT_VERSION, text, user, shape.model_json_schema()
        )
    for name, text, user, shape in (
        (
            "verifier: recover",
            verifier._VERIFY,
            verifier._USER_VERIFY,
            verifier._Recovered,
        ),
        (
            "verifier: supported",
            verifier._SUPPORT,
            verifier._USER_JUDGE,
            verifier._Supported,
        ),
        (
            "verifier: computes",
            verifier._COMPUTES,
            verifier._USER_JUDGE,
            verifier._Supported,
        ),
        (
            "verifier: follows",
            verifier._FOLLOWS,
            verifier._USER_JUDGE,
            verifier._Supported,
        ),
    ):
        yield Composed(
            name, verifier.PROMPT_VERSION, text, user, shape.model_json_schema()
        )
