"""The model-backed extractor for the claims a passage carries."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from extraction.extractors.base import ExtractionFailed, Extractor
from extraction.models import CandidateFact, PassageToExtract, Provenance
from llm.client import Client, ModelUnavailable

#: Recorded on every fact drawn with the prompt below. Bumped whenever that
#: prompt changes what counts as a fact.
PROMPT_VERSION = "8"

_SYSTEM = """ROLE
You are a claims analyst. You read one numbered excerpt of a document and
separate out the individual claims it makes.

ACTION
Break the excerpt down into the separate claims it makes. Each claim becomes
one fact, written by you and citing the numbered sentence you read it in.

STEPS
1. Read the section context, then the excerpt.
2. Go through it CLAIM BY CLAIM, not sentence by sentence. One sentence
   usually carries several claims, and each of them is its own fact:

     "The device weighs 4 kg, runs for 12 hours and ships in March."
       -> three facts: the weight, the runtime, the shipping month.

3. Drop anything that states nothing: a question, a checklist item, a heading
   written as a question, a sentence with no factual content.
4. For each claim that is left, note WHICH numbered sentence it came from.
5. Write that claim as one short sentence of your own, naming its subject.
6. Check every number, date, name and amount you wrote against the sentence
   you cited, and drop whatever that sentence does not carry.

CONTEXT
You are given the section context the excerpt sits under, and the excerpt
itself with its sentences numbered from zero. The context says what the
excerpt is about and is NOT part of it, so no sentence number belongs to it.
You have no knowledge beyond what you are shown and may use none.

EXAMPLES
Excerpt, under the context "Support > Response times":

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

A SENTENCE THAT ASKS SOMETHING STATES NOTHING, and is not to be turned round
into the claim it would be if the answer were yes:

  "Are imported data checked for validity?"
    -> no fact. The excerpt asks this; it does not say it happens.

FORMAT
Return `facts`, a list in which every entry has two parts:

- `sentences` is the NUMBER of the sentence the claim comes from, as a list.
  Copy nothing; just say which numbered sentence you read it in. Use two
  numbers only when the claim genuinely needs both. Several facts may cite
  the same sentence, which is normal and correct.
- `statement` is WRITTEN BY YOU: one short sentence carrying ONE claim, which
  a reader who cannot see the excerpt understands on its own.

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
- A sentence with no factual content yields no facts. An empty list is a
  correct answer and is better than a weak one.
"""

#: The user message, as the catalogue records it and `_prompt` renders it.
#: One source, so the recorded template and the sent text cannot drift.
_USER = """{{context}}Excerpt:
{{excerpt}}"""

#: Appended when EXTRACTION_MIN_OTHER_SHARE caps how many atomic facts a
#: passage keeps. Asked for here as well as enforced on what comes back,
#: because the facts over the cap are thrown away and a model told to write
#: four instead of nineteen spends a fifth of the tokens reaching the same
#: set. Left off entirely when there is no cap, rather than written as a
#: large number: a ceiling nothing is near still changes what a model writes.
_CAP = """
MOST IMPORTANT RULE, ABOVE ALL THE OTHERS:

- AT MOST {cap} FACTS from this excerpt, however long it is. Returning fewer
  is correct; returning more is not.
- CHOOSE THE {cap} SOMEBODY WOULD LOOK UP. A claim carrying a number, a date,
  a name, a limit, a duty or a definition comes first. Boilerplate - who holds
  a copyright, who reviewed the document, which edition a thing belongs to -
  is left out even when it is the only thing the excerpt says.
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


def composed(cap: int | None = None) -> str:
    """This extractor's system prompt, as it is sent under one cap.

    A function because two callers need the same text and neither may
    rebuild it: the extractor, which sends it, and `extraction.prompts`,
    which records what was sent. Written twice, a record of the uncapped
    text under a capped run would be a record of something no model saw.
    """
    return _SYSTEM + _CAP.format(cap=cap) if cap else _SYSTEM


class LlmExtractor(Extractor):
    """Reads a passage's claims one at a time with a local model.

    The fallback for every passage the deterministic readers do not claim.
    The model sees one passage's numbered sentences and its heading trail,
    never the whole document, and cites a sentence by number rather than
    quoting it.
    """

    block_types: ClassVar[tuple[str, ...]] = ()
    method: ClassVar[str] = "llm"

    def __init__(self, client: Client, cap: int | None = None) -> None:
        """Initialises the extractor with the model it asks.

        Args:
            client: The model that reads a passage.
            cap: The most facts one passage may yield, or None for no
                ceiling. What EXTRACTION_MIN_OTHER_SHARE works out to.
        """
        self._client = client
        self._system = composed(cap)

    @property
    def provenance(self) -> Provenance:
        """What produced these facts, recorded on each one."""
        return Provenance(
            model=self._client.model,
            prompt_version=PROMPT_VERSION,
            temperature=self._client.temperature,
        )

    def extract(self, passage: PassageToExtract) -> list[CandidateFact]:
        """Asks the model for the claims in one passage.

        Args:
            passage: The passage to read.

        Returns:
            One atomic candidate per claim, each citing the sentences it was
            drawn from. Empty when the passage carries none.

        Raises:
            ExtractionFailed: If the model could not be reached.
        """
        try:
            answer = self._client.answer(
                system=self._system,
                user=self._prompt(passage),
                shape=_Facts,
                prompt_version=PROMPT_VERSION,
            )
        except ModelUnavailable as exc:
            raise ExtractionFailed(str(exc)) from exc
        return [
            CandidateFact(
                statement=fact.statement, sentences=tuple(dict.fromkeys(fact.sentences))
            )
            for fact in answer.facts
            if fact.statement.strip() and fact.sentences
        ]

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
        return _USER.replace("{{context}}", heading).replace("{{excerpt}}", numbered)
