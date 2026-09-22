"""The kinds of question that can be asked, and what each one asks for.

Question form, never subject matter: every type here is askable of a manual,
a contract, a report or a policy, and nothing in the prompts names a domain.

One shared rule block holds what is true of every question - do not name the
source, name the subject, one question, the language of the facts - and each
type adds what it asks for, what its answer looks like, and one worked
example. Two prompts for one rule is how the two come to disagree.

`form` decides which gates apply: a `value` may carry no verb, an
`explanation` must, and each has its own length bounds.

`passages` is the fewest distinct passages a sample must offer for the type
to be writable at all. A comparison drawn from one passage is a question
about one thing.
"""

from __future__ import annotations

from dataclasses import dataclass

from database.qa_generator import (
    AnswerForm,
    CognitiveLevel,
    Derivation,
    Difficulty,
    QuestionType,
)

#: Recorded in the log beside every question written with the prompts below.
#: Bumped whenever one changes what a question is: two prompts are two
#: datasets, as with extraction.
PROMPT_VERSION = "8"

#: How every question must READ, whatever it is for. Shared by the writer and
#: by the perturbation that writes the unanswerable ones: a rule in one prompt
#: and not the other is how the two come to disagree, and it showed - every
#: question padded with `spezifische` was an unanswerable one, written by the
#: only prompt that had never been told not to.
READS = """Write the question somebody who needs this information would actually
type. They have not read the passage. They do not know which document answers
them - finding that out is the whole reason they are asking.

Rules, all of them mandatory:

- NEVER SAY WHERE THE ANSWER IS. No "according to the report", no "in this
  circular", no naming or quoting a document, a section or a heading. Nobody
  asks a service desk a question while telling it which file to open. A
  question that cites its own source has already done the work it was meant to
  test, and it is thrown away.

- NAME THE SUBJECT, NOT THE SOURCE. The subject is what the question is about:
  the thing, the party, the duty, the period. The source is which document
  says it. Name the first, never the second. A question opening with a bare
  "What" or "Which" that names nothing at all is no good either.

- DO NOT HAND THE FACT BACK. Taking the sentence and replacing one part with a
  question word is the failure this task is about. Ask what a person would ask.

- POINT AT NOTHING THE ASKER CANNOT SEE. No "this", "these", "that one", "the
  said", "the two above", "the aforementioned" - the reader has no passage in
  front of them, so a word pointing outside the question points at nothing.
  Name the thing instead: "between these two syllabi" is unanswerable, "between
  the 2022 and 2024 editions" is a question. This applies to every question
  asked cold; only a follow-up in a conversation may lean on what came before.

- NO PADDING WORDS. Nobody typing a question into a search box writes the
  equivalent of "specific", "concrete" or "particular" - in any language they
  are filler that makes a question read like a form. Ask "Which criteria ...",
  never "Which specific criteria ...". Say the thing plainly.

- VARY HOW YOU OPEN. A kind of question has more than one opening and real
  askers use all of them: a reason is asked as "Why ...", "What is the reason
  ...", "What makes ... necessary"; a comparison as "How do ... differ",
  "What sets ... apart", "Which of ... is". Where a type below shows two
  worked examples, they open differently ON PURPOSE - take the variety, not
  just the first one. A set in which nine of ten questions of a kind open
  with the same word tests whether a chatbot handles that word, not whether
  it can find an answer.

- NEVER PUT THE ANSWER IN THE QUESTION, or the word the answer is a kind of.

- ONE question, ending in a question mark. One thing asked.

- Write in the language of the facts.
"""

#: What the second answer is for. Every question carries two, and they are
#: scored by different gates because they make different claims: the target
#: is matched against what a verifier independently recovered, and this one
#: is only ever checked for asserting something the passages do not.
#:
#: Written as a rule AND into every worked example below, because the
#: examples are what a model actually copies. An instruction asking for
#: prose, under an example answering `it is closed`, gets `it is closed`.
EXPLAINS = """You write TWO answers to your own question, and they are not the
same answer written twice.

`answer` is the KEY. It is what a chatbot's reply is matched against, so it
stays exactly as short and as literal as the form below demands. Do not pad it,
do not explain in it, do not add a clause to make it read better.

`explanation` is what somebody who has never seen this material needs in order
to understand the answer. Write it for them:

- THREE TO SIX SENTENCES. Prose, not bullets, not a heading.

- SAY WHY, NOT JUST WHAT. The key already says what. This says what the thing
  is, how it works, what it is for, what follows from it, what it is in
  contrast to - whatever the material actually gives you. An explanation that
  restates the key in longer words has explained nothing.

- STAND ON ITS OWN. Somebody reads the question and this, and nothing else.
  Name the thing again rather than saying "it"; give the context the passage
  gives you.

- ONLY WHAT THE FACTS AND PASSAGES SAY. No outside knowledge, no filling a gap
  with what is usually true. Every number, date and name in it has to be in the
  material in front of you - one that is not is how this gets thrown away.

- SAME LANGUAGE as the question, and NEVER SAY WHERE THE ANSWER IS: the rules
  above about naming a document, a section or a heading apply here too.
"""

_RULES = f"""You write ONE test question for measuring a document-search chatbot.

You are given numbered FACTS drawn from a corpus, and the PASSAGE each came
from. The question must be answered by the facts. The passage is there so you
know what the material is about - use it to phrase the question, never as
something to ask about and never as something to cite.

{READS}
- `facts` is the NUMBERS of the facts your question needs. Use several only
  when the question genuinely cannot be answered without all of them.

{EXPLAINS}"""

#: Appended to the rules above for every type whose sample spans more than one
#: passage. Without it the writer answers the first passage and ignores the
#: rest, and the row carries a spread the question never used.
_SPAN = """
- THIS QUESTION SHOULD NEED FACTS FROM MORE THAN ONE PASSAGE, so that a
  chatbot has to find and combine two places in the material rather than one.

- BUT IT IS STILL ONE QUESTION ABOUT ONE THING, and that comes first. One
  question word, one thing asked. Do NOT weld two questions together with
  "and":

    WRONG  "Which steps prepare a site AND how high is the fee for a permit?"
    WRONG  "Why did the backlog ease in 2025 AND how is cover assessed?"
           (two questions in a trenchcoat; a chatbot answering one of them is
            neither right nor wrong, and nobody types this)

    RIGHT  "How many inspections were carried out across the northern and
            southern sites in total?"
    RIGHT  "How do the reply times for standard and urgent requests differ?"
           (one thing asked, which happens to need both facts)

- THE FACTS MUST MEET. Two facts belong in one question only when they are
  about the same thing: the same subject seen twice, one naming what the
  other defines, two values of one measure, two parties under one duty. Two
  facts that merely arrived together do NOT meet, and no question spans them
  honestly.

    WRONG  "How do X and Y differ in category and date of first use?"
           (one fact gives a category, the other a date; there is no
            comparison here, only two unrelated facts forced into one shape)

- TWO DOCUMENTS ARE TWO SOURCES. Where the passages are marked `Document A`
  and `Document B`, they are different material and may simply disagree. A
  claim from one is not a claim about the other, and the two do not average.

    WRONG  "How many chapters with examinable content does the syllabus
            have?" answered "once three, once eight"
           (two documents answering differently is not one answer; the
            question has no single truth and nobody can be marked on it)

    RIGHT  "How do the two syllabi differ in the number of chapters with
            examinable content?"
           (asks about the difference, which IS one thing, and is true)
    RIGHT  ask about one document and cite only its fact

- IF THE FACTS HAVE NO SINGLE HONEST QUESTION BETWEEN THEM, ask about one of
  them alone and name only the facts you used. THIS IS THE EXPECTED ANSWER,
  not a failure: a narrower question somebody would actually type beats a
  wide one nobody would, and citing one fact of three is a correct outcome.
"""

_FORMS: dict[str, str] = {
    AnswerForm.VALUE: (
        "A SHORT NOUN PHRASE, a few words at most: a value, an amount, a date, "
        "a name, a limit, a share. Never a sentence, never a clause, never "
        "anything with a verb in it. If the answer you want to write is a "
        "sentence, you asked too broad a question."
    ),
    AnswerForm.LIST: (
        "THE ITEMS THEMSELVES, separated by commas or by `and`. Short phrases, "
        "not prose and not a sentence about them. Two to six of them. If there "
        "is only one item, you asked the wrong question for this kind."
    ),
    AnswerForm.EXPLANATION: (
        "ONE OR TWO SENTENCES that actually answer it, in the words of the "
        "material. Prose, not a noun phrase and not a heading. It has to say "
        "something a reader could mark right or wrong."
    ),
}


@dataclass(frozen=True)
class TypeSpec:
    """One kind of question: what it asks for and what its answer is."""

    name: str
    form: str
    #: The fewest distinct passages a sample must offer for this to be written.
    passages: int
    #: One line, shown to the writer as the task and to the verifier as what
    #: the question was supposed to do.
    asks: str
    #: What to ask, and one worked example, appended to the shared rules.
    directive: str
    #: The other forms this type's answer is allowed to come back as. The
    #: prompt still asks for `form`; these are what the gates will also take.
    #:
    #: One form per type was refusing good answers. The material decides what
    #: shape an answer has, not the question: `Welche Werkzeuge werden
    #: empfohlen?` is a factoid whose answer is three tools, and a `condition`
    #: is answered `immer` when that is the condition. Both were thrown out
    #: for the form they arrived in rather than for being wrong, and the
    #: stricter comparison a `value` is held to then refused them twice.
    also: tuple[str, ...] = ()
    #: How an answer that is not stated is got out of the material, or None
    #: when the material states it outright. It decides which gate the
    #: question faces: recoverability asks whether the passages STATE the
    #: answer, and for a derived type that is the wrong question - the whole
    #: point is that the answer is not there to be found.
    derived: str | None = None
    #: How much the question asks of whoever answers it. Declared by the
    #: type rather than judged per question, so two readers cannot disagree,
    #: and stored beside `difficulty`, which measures something else: how
    #: far the answer is spread, not what has to be done with it.
    level: str = CognitiveLevel.RECALL
    #: The lowest band this type may be planned at, when needing more than
    #: one passage is not the reason. Only a type that needs ONE and is
    #: still not a lookup has to say so; see `floor`.
    least: str | None = None

    @property
    def spans(self) -> bool:
        """Whether this type needs facts from more than one passage."""
        return self.passages > 1

    @property
    def floor(self) -> str:
        """The lowest band this type may be planned at.

        A band is a request for the SHAPE of a sample - `easy` offers one
        passage, and that is the whole of what the band means - so a type
        needing two cannot be planned easy whatever else is true of it.

        That is not the only reason a type is not easy, and conflating the
        two was costing `application` half its slots. It needs one passage,
        because the rule it applies sits in one; it is still not a lookup,
        because the case it puts that rule to is not in the material. The
        same distinction `cognitive_level` draws against `difficulty`:
        how far the answer is spread is not how much work it takes.
        """
        if self.least:
            return self.least
        return Difficulty.MEDIUM if self.spans else Difficulty.EASY

    @property
    def forms(self) -> tuple[str, ...]:
        """Every form this type's answer may take, the asked-for one first."""
        return (self.form, *self.also)

    @property
    def answer_rule(self) -> str:
        """What this type's answer has to look like, for a prompt."""
        return _FORMS[self.form]

    def system(self, *, spans: bool = False) -> str:
        """The whole system prompt for writing a question of this type.

        `spans` adds the instruction to use facts from more than one passage.
        Set by the plan from the shape of the sample, not by the type: a
        factoid drawn from two documents is a cross-document factoid.
        """
        return (
            f"{_RULES}{_SPAN if spans or self.spans else ''}\n"
            f"WHAT TO ASK: {self.asks}\n\n"
            f"THE ANSWER IS {_FORMS[self.form]}\n\n"
            f"{self.directive}"
        )


_SPECS = (
    TypeSpec(
        name=QuestionType.FACTOID,
        level=CognitiveLevel.RECALL,
        form=AnswerForm.VALUE,
        passages=1,
        also=(AnswerForm.LIST,),
        asks="one checkable value - how many, how much, by when, what limit",
        directive="""Worked example. Facts:

  [1] A standard support request is answered within 48 hours.
  [2] An urgent support request is answered within 4 hours.

  WRONG  "Within how many hours is a standard request answered?"
         (the fact with its number deleted; nobody types this)
  WRONG  "What are the response times?"
         (names nothing, and no short answer is right)

  RIGHT  question: "How long is allowed for answering a standard support
                    request?"
         answer:   "48 hours"
         explanation: "A standard support request has to be answered within 48
                    hours. That is the ordinary service level, and it is the
                    slower of the two the material sets: a request marked
                    urgent is answered within 4 hours instead. The 48 hours are
                    the time allowed for the answer, not for resolving what was
                    asked about."
         facts:    [1]
""",
    ),
    TypeSpec(
        name=QuestionType.ENTITY,
        level=CognitiveLevel.RECALL,
        form=AnswerForm.VALUE,
        passages=1,
        also=(AnswerForm.LIST,),
        asks="who or which party does, decides, owns or must be told something",
        directive="""Worked example. Facts:

  [1] The site manager approves every change to the shift plan.

  WRONG  "Who approves it?"            (names nothing)

  RIGHT  question: "Who signs off a change to the shift plan?"
         answer:   "the site manager"
         explanation: "Every change to the shift plan is approved by the site
                    manager. Approval rests with that one role rather than with
                    whoever proposed the change, so a shift plan cannot be
                    altered by the team working to it. The requirement covers
                    every change, not only the substantial ones."
         facts:    [1]
""",
    ),
    TypeSpec(
        name=QuestionType.DEFINITION,
        level=CognitiveLevel.UNDERSTAND,
        form=AnswerForm.EXPLANATION,
        passages=1,
        also=(AnswerForm.VALUE,),
        asks="what a named thing, term or status IS, as the material defines it",
        directive="""Ask about a term the material itself defines or describes. Do
not ask for a dictionary definition of an ordinary word.

Worked example. Facts:

  [1] A priority request is one raised by phone and confirmed in writing.

  WRONG  "What is a request?"          (not defined here, and too broad)

  RIGHT  question: "What counts as a priority request?"
         answer:   "one that is raised by phone and confirmed in writing"
         explanation: "A priority request is defined by how it is raised rather
                    than by what it asks for. Two things have to happen: it is
                    raised by phone, and it is then confirmed in writing. Both
                    are required, so a phone call nobody confirms is not a
                    priority request and neither is a written request that
                    began as an email. The written confirmation is what leaves
                    a record of it."
         facts:    [1]
""",
    ),
    TypeSpec(
        name=QuestionType.ENUMERATION,
        level=CognitiveLevel.UNDERSTAND,
        form=AnswerForm.LIST,
        passages=1,
        asks="which things belong to a named set - the items, not how many",
        directive="""There has to be a real set in the facts. If the material lists
one thing, this is the wrong kind of question for it.

Worked example. Facts:

  [1] A request may be raised by phone.
  [2] A request may be raised through the web form.
  [3] A request may be raised by email.

  WRONG  "How many ways can a request be raised?"   (that is a factoid)

  RIGHT  question: "Which ways can a support request be raised?"
         answer:   "by phone, through the web form and by email"
         explanation: "There are three routes into the support process: a phone
                    call, the web form, and email. They are alternatives rather
                    than steps, so a request needs only one of them. Which one
                    is used still matters elsewhere, because a request raised
                    by phone is the one that can become a priority request once
                    it is confirmed in writing."
         facts:    [1, 2, 3]
""",
    ),
    TypeSpec(
        name=QuestionType.CONDITION,
        level=CognitiveLevel.APPLY,
        form=AnswerForm.LIST,
        passages=1,
        also=(AnswerForm.VALUE,),
        asks="when, or under what circumstances, something applies or is required",
        directive="""Ask for the circumstances, not for the thing itself.

Worked example. Facts:

  [1] The four-hour reply time applies only to requests marked urgent.
  [2] The four-hour reply time applies only on working days.

  WRONG  "How quickly is an urgent request answered?"   (that is a factoid)

  RIGHT  question: "When does the four-hour reply time apply?"
         answer:   "for requests marked urgent, and only on working days"
         explanation: "The four-hour reply time is not the general rule. Two
                    conditions have to hold together: the request must be
                    marked urgent, and the clock only runs on working days.
                    A request that is not marked urgent falls back to the
                    ordinary time, and an urgent one raised outside working
                    days does not consume its four hours until the next
                    working day begins."
         facts:    [1, 2]

  A second, opening differently:

  RIGHT  question: "What has to be true for a support request to get the
                    four-hour reply time?"
         answer:   "it is marked urgent, and the time runs on working days only"
         explanation: "Both conditions have to hold. The request has to carry
                    the urgent marking, and the four hours are counted in
                    working days, so one raised on a Friday evening does not
                    start consuming them until the Monday. Without the
                    marking the ordinary time applies instead."
         facts:    [1, 2]
""",
    ),
    TypeSpec(
        name=QuestionType.REASON,
        level=CognitiveLevel.ANALYSE,
        form=AnswerForm.EXPLANATION,
        passages=1,
        asks="why something is required, done, or the way it is",
        directive="""Only where the material actually gives a reason. If it states a
rule and never says why, this is the wrong kind of question for it - do not
invent the reason.

Worked example. Facts:

  [1] Requests are confirmed in writing so the agreed time can be evidenced.

  WRONG  "Why is writing important?"   (not what the material says)

  RIGHT  question: "Why does a request have to be confirmed in writing?"
         answer:   "so that the agreed response time can be evidenced later"
         explanation: "The written confirmation exists to create evidence. A
                    request agreed by phone leaves nothing showing what was
                    promised, so if the response time is disputed afterwards
                    there is no record to settle it. Confirming in writing
                    fixes the agreed time at the moment it is agreed, which is
                    what makes the commitment enforceable rather than merely
                    stated."
         facts:    [1]

  A second, opening differently. Both are good questions; 92% of one measured
  run opened with the first word above, which is a set testing one word.

  RIGHT  question: "What makes a written confirmation necessary for a
                    support request?"
         answer:   "it is what evidences the agreed response time later"
         explanation: "The requirement exists so that the agreed time can be
                    evidenced. Without a written record a disputed response
                    time comes down to what two people remember, and the
                    commitment cannot be held to. The confirmation fixes it
                    in writing at the point it is agreed."
         facts:    [1]
""",
    ),
    TypeSpec(
        name=QuestionType.PROCEDURE,
        level=CognitiveLevel.APPLY,
        form=AnswerForm.EXPLANATION,
        passages=1,
        also=(AnswerForm.LIST,),
        asks="how something is done, or in what order the steps go",
        directive="""Ask for the method. The answer names the steps, in order, in
the words of the material.

Worked example. Facts:

  [1] A request is raised through the web form.
  [2] The form is confirmed by email before work begins.

  RIGHT  question: "How is a support request raised and confirmed?"
         answer:   "it is raised through the web form, and confirmed by email
                    before work begins"
         explanation: "The process runs in two steps and the order is fixed.
                    The request is first raised through the web form, which is
                    what puts it into the system. It is then confirmed by
                    email, and that confirmation has to arrive before any work
                    starts. The sequencing is the point: work begun on an
                    unconfirmed request has no agreed scope behind it."
         facts:    [1, 2]
""",
    ),
    TypeSpec(
        name=QuestionType.CONSEQUENCE,
        level=CognitiveLevel.ANALYSE,
        form=AnswerForm.EXPLANATION,
        passages=1,
        asks="what happens, or what follows, when something is or is not done",
        directive="""Only where the material states the consequence. Do not invent
one.

Worked example. Facts:

  [1] A request left unconfirmed for five working days is closed.

  RIGHT  question: "What happens to a request nobody confirms for five working
                    days?"
         answer:   "it is closed"
         explanation: "An unconfirmed request does not stay open indefinitely.
                    Once five working days pass without confirmation it is
                    closed. The count is in working days rather than calendar
                    days, so a weekend does not shorten the window. Closing is
                    automatic in the sense that it follows from the time
                    elapsing, not from anybody deciding the request is no
                    longer wanted."
         facts:    [1]
""",
    ),
    TypeSpec(
        name=QuestionType.COMPARISON,
        level=CognitiveLevel.ANALYSE,
        form=AnswerForm.LIST,
        passages=2,
        also=(AnswerForm.EXPLANATION,),
        asks="how two named things differ, as the material states each of them",
        directive="""Both sides must come from the facts. Name both in the question.

Worked example. Facts:

  [1] A standard support request is answered within 48 hours.
  [2] An urgent support request is answered within 4 hours.

  RIGHT  question: "How do the reply times for standard and urgent support
                    requests differ?"
         answer:   "48 hours for a standard request and 4 hours for an urgent
                    one"
         explanation: "The two classes of request carry different commitments.
                    A standard request is answered within 48 hours; an urgent
                    one within 4. The urgent time is twelve times shorter, so
                    marking a request urgent is what determines whether it is
                    handled the same working day or over the following two.
                    Both figures are times to answer, so the two are directly
                    comparable."
         facts:    [1, 2]

  A second, opening differently:

  RIGHT  question: "What sets the reply time for an urgent support request
                    apart from a standard one?"
         answer:   "4 hours rather than 48"
         explanation: "An urgent request is answered within 4 hours where a
                    standard one has 48. The difference is what marking a
                    request urgent buys: the same working day rather than the
                    following two. Both are times to answer rather than times
                    to resolve, so they measure the same thing."
         facts:    [1, 2]
""",
    ),
    TypeSpec(
        name=QuestionType.AGGREGATION,
        level=CognitiveLevel.ANALYSE,
        form=AnswerForm.VALUE,
        passages=2,
        derived=Derivation.ARITHMETIC,
        asks="a total, a count or a sum that no single fact states on its own",
        directive="""The answer must be something the material does not write down
anywhere - it has to be worked out from two or more facts. If one fact already
states it, this is the wrong kind of question.

THE FIGURES MUST BE THE SAME KIND OF THING, counted in the same unit, and
their total must be a quantity somebody would want. Two numbers that merely
sit in the same corpus do not add up to anything: a year is not a count, a
version number is not an amount, and a page number is not a duration. If the
facts offer no two figures that measure one thing between them, this is the
wrong kind of question for them - say so by asking about one of them instead.

Worked example. Facts:

  [1] The northern site employs 40 people.
  [2] The southern site employs 25 people.

  RIGHT  question: "How many people do the northern and southern sites employ
                    between them?"
         answer:   "65"
         explanation: "The two sites employ 65 people in total. The material
                    gives the figures separately - 40 at the northern site and
                    25 at the southern one - and states no combined number
                    anywhere, so the total has to be added up. Both figures
                    count the same thing, employees at a site, which is what
                    makes them addable."
         facts:    [1, 2]

  WRONG  facts:    [1] A method was first described in 2011.
                   [2] The prior edition carried the version year 2023.
         question: "What year do the first description and the prior edition
                    come to together?"
         answer:   "4034"
         (two years are not two counts of one thing; their sum measures
          nothing and no reader would ask for it)
""",
    ),
    TypeSpec(
        name=QuestionType.TEMPORAL,
        level=CognitiveLevel.ANALYSE,
        form=AnswerForm.LIST,
        passages=2,
        also=(AnswerForm.EXPLANATION,),
        asks="what changed between two periods, or what the order of events was",
        directive="""Both periods, or both events, must be in the facts.

Worked example. Facts:

  [1] The reply time was 72 hours in 2024.
  [2] The reply time was 48 hours in 2025.

  RIGHT  question: "How did the reply time for a standard request change from
                    2024 to 2025?"
         answer:   "from 72 hours in 2024 to 48 hours in 2025"
         explanation: "The commitment for a standard request tightened between
                    the two years. In 2024 it stood at 72 hours and in 2025 at
                    48, a reduction of a full day. Both figures describe the
                    same class of request, so the change is a real tightening
                    of the service level rather than a difference in what was
                    being measured."
         facts:    [1, 2]
""",
    ),
    TypeSpec(
        name=QuestionType.IMPLICATION,
        level=CognitiveLevel.ANALYSE,
        derived=Derivation.ENTAILMENT,
        form=AnswerForm.EXPLANATION,
        passages=2,
        asks="what must be true when two things the material states both hold",
        directive="""Both premises must be in the facts, and the conclusion must
be in NEITHER. If the material already says it, that is a `consequence` and not
this: what makes this kind worth asking is that somebody has to put two
statements together and see what they come to.

Do not invent a premise. If the facts do not settle the question between them,
there is no implication here to ask about.

Worked example. Facts:

  [1] A standard request is answered within 48 hours.
  [2] The 48-hour time counts working days only.

  WRONG  "How long is allowed for a standard request?"
         (that is fact [1] with its number deleted; nobody has to reason)

  RIGHT  question: "What is the latest a request raised on a Friday can be
                    answered?"
         answer:   "the following Tuesday, because the 48 hours count only
                    working days and the weekend does not"
         explanation: "Two rules combine here. A standard request is answered
                    within 48 hours, and those 48 hours count working days
                    only. A request raised on a Friday therefore has one
                    working day left that week; the Saturday and Sunday are not
                    counted, and the remaining day falls on the Monday. The
                    deadline lands at the end of the Tuesday. The material
                    states neither the Friday case nor the Tuesday answer - it
                    follows from putting the two rules together."
         facts:    [1, 2]
""",
    ),
    TypeSpec(
        name=QuestionType.APPLICATION,
        level=CognitiveLevel.APPLY,
        derived=Derivation.ENTAILMENT,
        form=AnswerForm.EXPLANATION,
        also=(AnswerForm.VALUE,),
        passages=1,
        # One passage, because the rule it applies sits in one. Never easy,
        # because the case it puts that rule to is not in the material, so
        # answering is not a lookup however little of the corpus it reaches.
        least=Difficulty.MEDIUM,
        asks="which rule the material gives governs a case it does not mention",
        directive="""Put a CASE the material does not name to a rule it does.
The rule has to be in the facts; the case must not be, or there is nothing to
apply and the answer is a lookup.

The case must be one the rule actually settles. A case the material leaves open
is an unanswerable question, which is a different kind.

Worked example. Facts:

  [1] A request marked urgent is answered within 4 hours.
  [2] A request is marked urgent when it stops work at a site.

  WRONG  "When is a request marked urgent?"
         (that is fact [2] read back; no case, no application)

  RIGHT  question: "A site cannot dispatch because its scanner is down and a
                    request is raised. How quickly must it be answered?"
         answer:   "within 4 hours, because work has stopped at the site and
                    that is what marks a request urgent"
         explanation: "The rule to apply is the one about what makes a request
                    urgent: a request is marked urgent when it stops work at a
                    site. A scanner failure that prevents dispatch has stopped
                    work, so the request qualifies, and an urgent request is
                    answered within 4 hours rather than the standard 48. The
                    material never mentions scanners; the case is settled by
                    reading the stated rule onto it."
         facts:    [1, 2]

  A second, putting the case without opening on a conditional:

  RIGHT  question: "A delivery van breaks down and the depot stops loading.
                    How much time is allowed to answer the request that is
                    raised?"
         answer:   "4 hours"
         explanation: "A request counts as urgent when it stops work at a
                    site, and a depot that cannot load has stopped work. So
                    the urgent commitment applies and the answer is due
                    within 4 hours rather than the standard 48. Nothing in
                    the material mentions vans or depots - the case is
                    settled by applying the stated rule to it."
         facts:    [1, 2]
""",
    ),
)

#: Every type, by name.
SPECS: dict[str, TypeSpec] = {spec.name: spec for spec in _SPECS}

#: The types whose label a RULE settles, so a level derived from one is a
#: measurement rather than a name somebody wrote down.
#:
#: `cognitive_level` is derived from the type, and the type was unchecked
#: when the column was added: 208 of 244 accepted `entity` questions named
#: no party, which is the definition of the type. So the level inherited
#: whatever the label got wrong, and an exam blueprint reading the column
#: could not tell which rows to trust.
#:
#: Four of them now. `entity` and `enumeration` face a structural gate each;
#: `implication` and `application` are the derived types, whose entailment
#: and arithmetic checks ARE the reading that they reason rather than look
#: up. The other eight declare a level nothing has checked - `reason`,
#: `consequence` and `condition` have no structural signature and no model
#: is asked for one again. `/questions/quality` reports the share, so the
#: column says how much of itself is measured.
CHECKED: frozenset[str] = frozenset(
    {
        QuestionType.ENTITY,
        QuestionType.ENUMERATION,
        QuestionType.COMPARISON,
        QuestionType.TEMPORAL,
        QuestionType.IMPLICATION,
        QuestionType.APPLICATION,
    }
)


def spec(name: str | None) -> TypeSpec:
    """The spec one type name selects, falling back to the factoid.

    A stored row written before a type existed, or one whose type was dropped
    from the mix, still has to be re-checkable.
    """
    return SPECS.get(name or "", SPECS[QuestionType.FACTOID])
