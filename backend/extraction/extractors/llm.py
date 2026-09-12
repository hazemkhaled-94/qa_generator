"""The model-backed extractor for prose passages."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import instructor
import litellm
from instructor.core import InstructorRetryException
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from extraction.extractors.base import ExtractionFailed, Extractor
from extraction.models import CandidateFact, PassageToExtract, Provenance

log = logging.getLogger(__name__)

#: Failures worth another attempt. An authentication failure, an unknown
#: model or a malformed schema is none of these and is raised at once.
_TRANSIENT = (
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.ServiceUnavailableError,
    InstructorRetryException,
)

#: Recorded on every fact drawn with the prompt below. Bumped whenever that
#: prompt changes what counts as a fact: two prompts are two datasets.
PROMPT_VERSION = "5"

_SYSTEM = """You break a numbered excerpt down into the separate claims it
makes. Each claim becomes one fact.

Work claim by claim, not sentence by sentence. One sentence usually carries
several claims, and each of them is its own fact:

  "The device weighs 4 kg, runs for 12 hours and ships in March."
    -> three facts: the weight, the runtime, the shipping month.

A fact has two parts:
- `sentences` is the NUMBER of the sentence the claim comes from, as a list.
  Copy nothing; just say which numbered sentence you read it in. Use two
  numbers only when the claim genuinely needs both.
- `statement` is WRITTEN BY YOU: one short sentence carrying ONE claim, which
  a reader who cannot see the excerpt understands on its own.

Several facts may cite the same sentence. That is normal and correct.

Rules, all of them mandatory:
- ONE CLAIM per statement. Your statement must contain exactly one finite
  verb. If you can split it on "and", "as well as", or a comma joining two
  facts, it is two facts.
- NAME THE SUBJECT. Never write "it", "this", "they", "he", "she". Replace
  every one with what the excerpt means, taking it from the section context
  when the sentence leaves it out.
- ADD NOTHING. Every number, date, name and amount in your statement must
  appear in the sentence you cite. Do not round, convert, infer or complete.
- Carry the qualifiers that make the claim true on its own: the date, the
  place, the party, the unit, the condition.
- A number, a date, a name, a limit, a duty and a definition are each a claim
  worth a fact of its own.
- Write the statement in the language of the excerpt.
- Do not copy the sentence out. A statement that repeats its sentence adds
  nothing; it has to carry ONE of the claims and leave the rest.
- A sentence with no factual content yields no facts. Returning none is a
  correct answer and is better than a weak one.

Worked example. Excerpt, under the context "Support > Response times":

  [0] Standard requests are answered within 48 hours on working days.
  [1] Urgent requests are answered within 4 hours and may be raised by phone.

  WRONG  statement: "Standard enquiries receive a reply within 48 hours on
                     business days, and urgent ones within 4 hours."
         (two claims in one sentence, and it restates rather than decomposes)

  RIGHT  four facts, each one claim:
    1. sentences: [0]
       statement: "A standard support request is answered within 48 hours."
    2. sentences: [0]
       statement: "The 48-hour reply time for standard support requests
                   counts working days only."
    3. sentences: [1]
       statement: "An urgent support request is answered within 4 hours."
    4. sentences: [1]
       statement: "An urgent support request may be raised by phone."
"""


class _Fact(BaseModel):
    """One fact as the model is asked to return it."""

    sentences: list[int] = Field(
        description="The number(s) of the excerpt sentence this claim comes "
        "from. Usually one."
    )
    statement: str = Field(
        description="ONE claim, written by you in one short sentence with "
        "exactly one finite verb, naming its subject and understandable "
        "without the excerpt."
    )


class _Facts(BaseModel):
    """The model's answer for one passage."""

    facts: list[_Fact] = Field(
        default_factory=list, description="Every fact in the excerpt; may be empty."
    )


def _mode(name: str) -> instructor.Mode:
    """Turns a mode name from the environment into the mode itself."""
    try:
        return instructor.Mode[name.strip().upper()]
    except KeyError:
        raise ValueError(
            f"EXTRACTION_STRUCTURED_MODE={name!r} is not an instructor mode; "
            f"expected one of {', '.join(sorted(m.name for m in instructor.Mode))}"
        ) from None


class LlmExtractor(Extractor):
    """Reads facts out of prose with a local model.

    The fallback for every passage the deterministic readers do not claim.
    The model sees one passage's numbered sentences and its heading trail,
    never the whole document, and cites a sentence by number rather than
    quoting it.
    """

    block_types: ClassVar[tuple[str, ...]] = ()
    method: ClassVar[str] = "llm"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        temperature: float,
        timeout: float,
        max_attempts: int,
        structured_mode: str = "JSON_SCHEMA",
    ) -> None:
        """Initialises the extractor and its client."""
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._timeout = timeout
        # Any: instructor replaces create() at run time, so a checker would
        # match these keywords against the unpatched signature.
        self._client: Any = instructor.from_litellm(
            litellm.completion, mode=_mode(structured_mode)
        )
        # Only _TRANSIENT: retrying a bad model name or schema costs the
        # backoff on every passage and buries the real error.
        self._attempt = retry(
            retry=retry_if_exception_type(_TRANSIENT),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._ask)

    @property
    def provenance(self) -> Provenance:
        """What produced these facts, recorded on each one."""
        return Provenance(
            model=self._model,
            prompt_version=PROMPT_VERSION,
            temperature=self._temperature,
        )

    def extract(self, passage: PassageToExtract) -> list[CandidateFact]:
        """Asks the model for the facts in one passage."""
        try:
            answer = self._attempt(passage)
        except Exception as exc:
            raise ExtractionFailed(f"{type(exc).__name__}: {exc}") from exc
        return [
            CandidateFact(
                statement=fact.statement, sentences=tuple(dict.fromkeys(fact.sentences))
            )
            for fact in answer.facts
            if fact.statement.strip() and fact.sentences
        ]

    def _ask(self, passage: PassageToExtract) -> _Facts:
        """Sends one passage to the model."""
        return self._client.chat.completions.create(
            model=self._model,
            # Omitted when unset: a provider with its own address would be
            # sent to the wrong one by a base URL meant for Ollama.
            **({"api_base": self._base_url} if self._base_url else {}),
            temperature=self._temperature,
            timeout=self._timeout,
            response_model=_Facts,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": self._prompt(passage)},
            ],
        )

    @staticmethod
    def _prompt(passage: PassageToExtract) -> str:
        """Builds the user message: the heading trail and numbered sentences.

        The heading is fenced off and labelled as context: run together with
        the excerpt, the model cited it as a sentence that does not exist.
        """
        heading = (
            f"Context (not part of the excerpt, use it only to name what the "
            f"excerpt is about): {passage.section_path}\n\n"
            if passage.section_path
            else ""
        )
        numbered = "\n".join(
            f"[{sentence.index}] {sentence.text}" for sentence in passage.sentences
        )
        return f"{heading}Excerpt:\n{numbered}"
