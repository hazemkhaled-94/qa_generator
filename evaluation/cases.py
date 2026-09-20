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
        # `target: None` makes this UNANSWERABLE, and a passage failing to
        # answer an unanswerable question is the point of writing one rather
        # than a fault in it - so the checker accepts it and is right to.
        # The case asserted the opposite and was measuring the gate for
        # answerable questions against a question that is not one.
        #
        # What it is here to catch is the answerable version: the same
        # plausible-but-absent question, carrying the answer somebody would
        # wrongly give it. That one must be refused, and by recoverability
        # rather than by anything structural.
        "name": "a plausible question the passage does not cover",
        "question": (
            "Within how many hours is a standard support request answered "
            "on a public holiday?"
        ),
        "passage": "Standard requests are answered within 48 hours on working days.",
        "target": "48 hours",
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


#: ── The three phrasing judgements ─────────────────────────────────────
#: What a correct reading of a question alone says about it, independent of
#: any passage. These exist because the verifier answers all three in the
#: same call that reads the answer out of the passages, and nothing measured
#: whether it answers them WELL: the only ground truth for
#: `names_its_source`, `self_contained` and `subject` was the model that
#: answered them, so a change to the prompt or the model moved the numbers
#: with nothing to say which way was better.
#:
#: Half of these are German and drawn from what this corpus actually
#: produced, rejections included. The hard cases are deliberately here:
#:
#: - a question naming a PARTY, a period or a regulated thing, which reads
#:   like a source and is not one - the failure the prompt spends six
#:   examples on;
#: - a pointing word whose antecedent is inside the question, which is
#:   self-contained however much it reads like a reference;
#: - a question that names nothing at all, whose subject is empty, against
#:   one whose subject is a bare noun that is still a subject.
#:
#: `subject` is scored on containment rather than on equality: what a correct
#: reading copies out is a span, and two correct readings disagree about its
#: edges. `holds` is a word the span must carry; "" means the question names
#: nothing and the span should be empty.
PHRASING = (
    # ── names its source ──────────────────────────────────────────────
    {
        "name": "names a section, in German",
        "question": (
            "Wie unterscheiden sich in Abschnitt 2.2 die Themen zur "
            "Testschätzung und zur Fehlerbehebung?"
        ),
        "language": "de",
        "names_its_source": True,
        "self_contained": True,
        "holds": "Testschätzung",
    },
    {
        "name": "names a chapter, in German",
        "question": "Welches Thema behandelte Kapitel 5 in der älteren Fassung?",
        "language": "de",
        "names_its_source": True,
        "self_contained": True,
        "holds": "Kapitel",
    },
    {
        "name": "names a bibliography entry, in German",
        "question": "In welchem Monat wurde die Referenz [R22] aufgerufen?",
        "language": "de",
        "names_its_source": True,
        "self_contained": True,
        "holds": "R22",
    },
    {
        "name": "names an author and a year, in German",
        "question": "Wie definiert Beck 2003 Refactoring in der testgetriebenen Entwicklung?",
        "language": "de",
        "names_its_source": True,
        "self_contained": True,
        "holds": "Refactoring",
    },
    {
        "name": "names a document, in English",
        "question": "According to the service agreement, how long may a reply take?",
        "language": "en",
        "names_its_source": True,
        "self_contained": True,
        "holds": "reply",
    },
    # ── names a party, a period or a thing, which is NOT a source ─────
    {
        "name": "names the party a duty falls on, not a source",
        "question": "How many faults were reported to the site manager in 2025?",
        "language": "en",
        "names_its_source": False,
        "self_contained": True,
        "holds": "faults",
    },
    {
        "name": "names a standard as the subject, not as a source",
        "question": "Welche Reviewverfahren beschreibt die Norm ISO/IEC 20246?",
        "language": "de",
        "names_its_source": True,
        "self_contained": True,
        "holds": "20246",
    },
    {
        "name": "names a regulated thing, in German",
        "question": "Wer trägt die Verantwortung für den vierteljährlichen Risikobericht?",
        "language": "de",
        "names_its_source": False,
        "self_contained": True,
        "holds": "Risikobericht",
    },
    {
        "name": "names a period, not a source",
        "question": "Within how many hours is an urgent support request answered?",
        "language": "en",
        "names_its_source": False,
        "self_contained": True,
        "holds": "support request",
    },
    # ── self-contained, and not ───────────────────────────────────────
    {
        "name": "points at the corpus itself, in German",
        "question": "Unter welchen Einteilungen werden Testverfahren in diesem Lehrplan klassifiziert?",
        "language": "de",
        "names_its_source": True,
        "self_contained": False,
        "holds": "Testverfahren",
    },
    {
        "name": "points at a context the asker cannot see, in German",
        "question": "Warum wird ISO/IEC/IEEE 29119-4 in diesem Zusammenhang erwähnt?",
        "language": "de",
        "names_its_source": False,
        "self_contained": False,
        "holds": "29119",
    },
    {
        "name": "points at unnamed angaben, in German",
        "question": "Wie unterscheiden sich der Lehrplaninhalt und ein Testteammitglied laut diesen Angaben?",
        "language": "de",
        "names_its_source": False,
        "self_contained": False,
        "holds": "Lehrplaninhalt",
    },
    {
        "name": "points at two editions it never names, in English",
        "question": "How long is the gap between these two editions?",
        "language": "en",
        "names_its_source": False,
        "self_contained": False,
        "holds": "editions",
    },
    {
        "name": "a pointing word answered inside the question",
        "question": (
            "If a system meets its target by editing the stored score instead "
            "of doing the task, how is this behaviour classified?"
        ),
        "language": "en",
        "names_its_source": False,
        "self_contained": True,
        "holds": "behaviour",
    },
    {
        "name": "a pointing word answered inside the question, in German",
        "question": (
            "Wenn ein Team lineare Skripterstellung einführt, ist dieser "
            "Ansatz für einen großen Umfang geeignet?"
        ),
        "language": "de",
        "names_its_source": False,
        "self_contained": True,
        "holds": "Skripterstellung",
    },
    {
        "name": "nothing points anywhere",
        "question": "How do the reply times for standard and urgent requests differ?",
        "language": "en",
        "names_its_source": False,
        "self_contained": True,
        "holds": "reply times",
    },
    # ── subject: named, and not ───────────────────────────────────────
    {
        "name": "names nothing at all, in English",
        "question": "What specific components are included?",
        "language": "en",
        "names_its_source": False,
        "self_contained": True,
        "holds": "",
    },
    {
        "name": "names nothing at all, in German",
        "question": "Für welche Kriterien gelten die Anforderungen?",
        "language": "de",
        "names_its_source": False,
        "self_contained": True,
        "holds": "",
    },
    {
        "name": "a subject a reader could not answer still exists",
        "question": "Wie viele Testfälle verlangt die Grenzwertanalyse bei drei Partitionen?",
        "language": "de",
        "names_its_source": False,
        "self_contained": True,
        "holds": "Grenzwertanalyse",
    },
    # ── The pointing a demonstrative does not carry ───────────────────────
    # The case the gate was written for and could not see. `diesen beiden`
    # is PronType=Dem and was caught; `den beiden` is an article plus
    # PronType=Ind and was not, so the measurement abstained and the verdict
    # never came. Both are here now, because a measurement that catches one
    # phrasing of a failure and not the other is the failure.
    # Both name plenty - which is the point. A question can name four things
    # and still be unanswerable because nothing says WHICH two it means, and
    # keeping these anchored is what stops them passing for naming cases.
    {
        "name": "points at two syllabi through a demonstrative, in German",
        "question": (
            "Wie groß ist der inhaltliche Unterschied zwischen diesen "
            "beiden Lehrplänen für Testanalysten?"
        ),
        "language": "de",
        "names_its_source": False,
        "self_contained": False,
        "holds": "Lehrplänen",
    },
    {
        "name": "points at two syllabi through a bare count, in German",
        "question": (
            "Wie groß ist der inhaltliche Unterschied zwischen den beiden "
            "Lehrplänen für Testanalysten?"
        ),
        "language": "de",
        "names_its_source": False,
        "self_contained": False,
        "holds": "Lehrplänen",
    },
    {
        "name": "counts two things it did name, in German",
        "question": (
            "Wie groß ist der Abstand zwischen der Fassung von 2022 und "
            "der Fassung von 2024?"
        ),
        "language": "de",
        "names_its_source": False,
        "self_contained": True,
        "holds": "2022",
    },
    {
        "name": "points through an anaphoric adjective, in English",
        "question": (
            "How do the aforementioned parties differ in their reporting duties?"
        ),
        "language": "en",
        "names_its_source": False,
        "self_contained": False,
        "holds": "parties",
    },
    # ── The two a rule must not guess at ──────────────────────────────────
    # One standard named as what states the answer, one named as the thing
    # being asked about. Same shape, opposite verdicts, so `cites_source`
    # abstains on both and whatever holds an opinion decides.
    {
        "name": "a standard cited as the source, in German",
        "question": "Welche Anforderungen stellt die Norm ISO/IEC 25010 an Wartbarkeit?",
        "language": "de",
        "names_its_source": True,
        "self_contained": True,
        "holds": "25010",
    },
    {
        "name": "an author named as the subject, in English",
        "question": "Which testing practice is Beck credited with introducing?",
        "language": "en",
        "names_its_source": False,
        "self_contained": True,
        "holds": "Beck",
    },
)
