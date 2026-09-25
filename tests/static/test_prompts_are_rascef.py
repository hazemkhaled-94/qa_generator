"""Every prompt this pipeline sends is RASCEF, and says so in its headings.

Role, Action, Steps, Context, Examples, Format. Twenty-five prompts across
five services were written to that shape and nothing held them to it: a
prompt is a string, and a rule added to one in a hurry goes wherever the
cursor was. A missing FORMAT is the one that bites - a model given no
answer shape invents one, and instructor retries it into a
`ModelUnavailable` that reads as the model being down.

The headings are checked, not the content. What a prompt SAYS is
`test_prompts_name_no_domain` and `test_prompts_pinned`; this is only that
the six sections are there to say it in.

Composed rather than as written. `_FOLLOW` carries no EXAMPLES of its own:
its worked example arrives with the type, through `{{type}}`, and checking
the template would report a gap that no call ever has.
"""

from __future__ import annotations

import pytest

from assessment.templates import TEMPLATES
from extraction.extractors import bridge, digest, llm
from question_generation import phrasing, verifier
from question_generation.generation import _FOLLOW, _PERTURB, _typed
from question_generation.types import SPECS
from topic_modelling import labels

#: The six sections, each its own line, in the order RASCEF names them.
SECTIONS = ("ROLE", "ACTION", "STEPS", "CONTEXT", "EXAMPLES", "FORMAT")


def _prompts() -> dict[str, str]:
    """Every system prompt this pipeline sends, composed as it is sent."""
    written = {
        f"questions/{name}": spec.system() for name, spec in sorted(SPECS.items())
    }
    written["questions/perturb"] = _PERTURB
    # Its example comes with the type, so one type stands for all thirteen:
    # what is being checked is the section, and `{{type}}` is what supplies
    # it whichever type is asked for.
    written["questions/follow"] = _FOLLOW.replace(
        "{{type}}", _typed(SPECS["condition"])
    )
    for name in ("_SOURCE", "_NAMES", "_CONTAINED"):
        written[f"phrasing/{name}"] = getattr(phrasing, name)
    for name in ("_VERIFY", "_SUPPORT", "_COMPUTES", "_FOLLOWS"):
        written[f"verifier/{name}"] = getattr(verifier, name)
    # Composed with an empty cap paragraph, which is what a deployment that
    # sets no EXTRACTION_MIN_OTHER_SHARE sends.
    written["extraction/atomic"] = llm.composed("")
    written["extraction/digest"] = digest._SYSTEM
    written["extraction/bridge"] = bridge._SYSTEM
    written["topics/label"] = labels._SYSTEM
    for kind, templates in TEMPLATES.items():
        for template in templates:
            written[f"assessment/{kind}.{template.metric}"] = template.system
    return written


PROMPTS = _prompts()


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_a_prompt_carries_every_rascef_section(name: str) -> None:
    """A section missing is a section the model was never given."""
    text = f"\n{PROMPTS[name]}"
    missing = [one for one in SECTIONS if f"\n{one}\n" not in text]

    assert not missing, (
        f"{name} is missing {', '.join(missing)}. Every prompt here is "
        f"RASCEF: {', '.join(SECTIONS)}, each on a line of its own."
    )


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_the_sections_come_in_the_order_rascef_names_them(name: str) -> None:
    """A FORMAT above its STEPS reads as two prompts spliced together."""
    text = f"\n{PROMPTS[name]}"
    at = [text.index(f"\n{one}\n") for one in SECTIONS]

    assert at == sorted(at), (
        f"{name} puts its sections in the order "
        f"{', '.join(one for _, one in sorted(zip(at, SECTIONS, strict=True)))}, and "
        f"RASCEF is {', '.join(SECTIONS)}."
    )


def test_every_prompt_the_pipeline_sends_is_covered_here() -> None:
    """The count, so a new prompt is added here rather than only shipped.

    Thirteen question types, the perturbation and the follow-up, three
    phrasing judgements, four verifier gates, three extraction readers, the
    topic labeller, and eleven assessment metrics.
    """
    assert len(PROMPTS) == 13 + 2 + 3 + 4 + 3 + 1 + 11
