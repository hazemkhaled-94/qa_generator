"""Every prompt the judge sends, for the record that resolves a version.

`assessments.prompt_version` named a version and nothing in the deployment
could read it back. The other four stages that call a model record theirs
at start-up; this phase did not, so a corpus judged under version 1 and one
judged under version 2 were the same seven columns with nothing saying what
either had been asked.

One version over all of them, unlike the stages upstream. `PROMPT_VERSION`
in `templates.py` covers the SET a kind is judged on as well as the wording
of each template, because a row's verdicts are read against both.

Read by `run.py` where it starts. Nothing here calls a model.
"""

from __future__ import annotations

from assessment.templates import PROMPT_VERSION, TEMPLATES
from stages.prompts import Composed

#: What the phase records its prompts under.
SERVICE = "assessment"


def catalogue() -> list[Composed]:
    """Every prompt this phase can send, composed as it would be sent.

    Both halves and the shape: the system prompt, the user message as its
    template, and the JSON schema the answer has to come back in.

    Named by kind AND metric, because `relevance` is asked of a fact, a
    topic and a question through three different templates. The metric
    alone would be one name over three prompts, and the table is unique on
    (service, version, name) - so two of the three would be written over
    the first and the record would say the judge asks one thing where it
    asks three.
    """
    return [
        Composed(
            f"{kind}: {one.metric}",
            PROMPT_VERSION,
            one.system,
            one.user,
            one.shape().model_json_schema(),
        )
        for kind, group in TEMPLATES.items()
        for one in group
    ]
