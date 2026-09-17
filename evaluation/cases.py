"""The golden cases, and nothing that runs them.

Held here rather than inside the eval tests because two things read them:
`make test-eval`, which prints what a served model scored, and
`make eval-experiment`, which puts the same cases in front of the same
model and records the scores in Phoenix so two runs can be compared.

One definition, so a case added for one is a case the other measures too.
A test module would have been the wrong home for the second reader: the
cases are a project asset - the accumulated record of what this pipeline
is supposed to get right - and several of them are here because they
caught something.

Data only. Nothing here imports a model, a checker or a gate, so reading
the cases costs nothing.
"""

from __future__ import annotations

from database.qa_generator import QuestionType

#: ── Extraction ────────────────────────────────────────────────────────
#: Passages and what a correct reading of each yields. `claims` is how many
#: separate facts the text carries, which is what a reader should find.
EXTRACTION = (
    {
        "text": (
            "The device weighs 4 kg and runs for 12 hours. "
            "It arrives in March 2026 from the Hamburg plant."
        ),
        "language": "en",
        "claims": 4,
    },
    {
        "text": (
            "Ein Risikobericht muss mindestens vierteljährlich erstellt werden. "
            "Die Geschäftsleitung trägt dafür die Verantwortung."
        ),
        "language": "de",
        "claims": 2,
    },
    {
        "text": (
            "Standard requests are answered within 48 hours on working days. "
            "Urgent requests are answered within 4 hours."
        ),
        "language": "en",
        "claims": 3,
    },
)


#: ── Question generation ───────────────────────────────────────────────
#: Questions whose verdict is not a matter of taste. Each is a question, the
#: passage it cites, and whether the answer is genuinely in that passage.
#:
#: The German case is the one that made the round trip the gate rather than
#: the similarity check it replaced. "Ein Liquiditätsmanagementtool ist eine
#: einjährige Rückgabefrist" survived an NLI model, an LLM judge and a
#: structural check, because it reads like its passage. It does not survive
#: being asked.
QUESTIONS = (
    {
        "name": "answered outright",
        "question": "Within how many hours is a standard support request answered?",
        "passage": (
            "Standard requests are answered within 48 hours on working days. "
            "Urgent requests are answered within 4 hours."
        ),
        "target": "48 hours",
        "language": "en",
        "recoverable": True,
    },
    {
        "name": "answered in other words",
        "question": "How quickly does an urgent support request get a reply?",
        "passage": (
            "Standard requests are answered within 48 hours on working days. "
            "Urgent requests are answered within 4 hours."
        ),
        "target": "4 hours",
        "language": "en",
        "recoverable": True,
    },
    {
        "name": "the LMT case: reads like the passage, is not in it",
        "question": "Was ist ein Liquiditätsmanagementtool?",
        "passage": (
            "Für den Fonds gilt eine einjährige Rückgabefrist. "
            "Die Verwaltungsgesellschaft veröffentlicht die Frist im "
            "Verkaufsprospekt."
        ),
        "target": "eine einjährige Rückgabefrist",
        "language": "de",
        "recoverable": False,
    },
    {
        "name": "a plausible question the passage does not cover",
        "question": (
            "Within how many hours is a standard support request answered "
            "on a public holiday?"
        ),
        "passage": "Standard requests are answered within 48 hours on working days.",
        "target": None,
        "language": "en",
        "recoverable": False,
    },
    {
        "name": "the right subject, the wrong number",
        "question": "Within how many hours is a standard support request answered?",
        "passage": "Standard requests are answered within 48 hours on working days.",
        "target": "4 hours",
        "language": "en",
        "recoverable": False,
    },
    # The kinds that could not be written at all before the answer form was a
    # column. Each of these was refused as malformed by the one verb rule.
    {
        "name": "a reason the material gives",
        "question": "Why does a support request have to be confirmed in writing?",
        "passage": (
            "Requests are confirmed in writing so that the agreed response "
            "time can be evidenced later. The confirmation is sent by email."
        ),
        "target": "so that the agreed response time can be evidenced later",
        "language": "en",
        "type": QuestionType.REASON,
        "recoverable": True,
    },
    {
        "name": "a reason the material does not give",
        "question": "Why is the reply time set at 48 hours rather than 24?",
        "passage": "Standard requests are answered within 48 hours on working days.",
        "target": "because two working days allow for a weekend backlog",
        "language": "en",
        "type": QuestionType.REASON,
        "recoverable": False,
    },
    {
        "name": "a procedure stated in order",
        "question": "How is a support request raised and confirmed?",
        "passage": (
            "A request is raised through the web form. The form is confirmed "
            "by email before any work begins."
        ),
        "target": (
            "it is raised through the web form and confirmed by email before "
            "work begins"
        ),
        "language": "en",
        "type": QuestionType.PROCEDURE,
        "recoverable": True,
    },
    {
        "name": "the items of a set",
        "question": "Which ways can a support request be raised?",
        "passage": (
            "A request may be raised by phone, through the web form or by "
            "email. Requests raised by phone are confirmed in writing."
        ),
        "target": "by phone, through the web form and by email",
        "language": "en",
        "type": QuestionType.ENUMERATION,
        "recoverable": True,
    },
    {
        "name": "a condition the material states",
        "question": "When does the four-hour reply time apply?",
        "passage": (
            "The four-hour reply time applies only to requests marked urgent, "
            "and only on working days."
        ),
        "target": "to requests marked urgent, and only on working days",
        "language": "en",
        "type": QuestionType.CONDITION,
        "recoverable": True,
    },
)
