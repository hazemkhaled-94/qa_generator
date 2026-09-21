# Question generation

Each topic's facts become questions with known answers, and each question
cites the facts it was written from.

This is the last stage, and the one the dataset is for. Everything before it
exists to put a verified, atomic, citable claim in front of a writer; this
turns that claim into something somebody might actually have typed into a
chatbot, and then tries hard to prove the result is no good.

A question that fails a gate is **stored with the gate's name** rather than
dropped. The rate at which that happens is how the writer is judged, so it
belongs in the data and not in a log.

## What a question is

A question is written from one or more **facts**, and the writer is shown the
**passages** they came from as well. The facts are what the answer must rest
on; the passage is what the question can be phrased from.

Both halves are load-bearing, and the first version of this stage had only
one. Shown a single atomic statement and nothing else, a model has one triple
to work with, so the only question available is that statement with one part
replaced by a question word — `Geopolitische Konflikte schüren Unsicherheit`
asked as `Was schüren geopolitische Konflikte?`. Nobody searching a corpus of
thousands of pages types that. The passage and its heading trail are where the
institution, the document and the period come from, which is what a question
has to name to be one somebody could have asked.

The unit of work is a **topic**, because a question's subject is one. The
facts a topic may be asked about are the facts of the passages that topic is
strongest in; a passage belongs a little to many topics, and writing a
question for every one of them asks the same thing under a dozen subjects. A
topic with `include_in_coverage = false` is skipped, which is the neutral
switch for "this is not a subject" — nothing in the code decides that.

## What it does

### Step 1 — The plan comes first

**Where:** [`planning.py`](planning.py).

Before anything is written, a topic gets a **plan**: one slot per question,
each carrying the kind of question to write, the difficulty band to aim for,
and whether it is meant to have an answer at all. Two settings decide it, and
both are proportions rather than counts:

```ini
QUESTIONS_TYPE_MIX=factoid:1,definition:1,entity:1,...
QUESTIONS_DIFFICULTY_MIX=easy:2,medium:3,hard:3
```

The weights are spread over the slots by **highest averages**, so the counts
are exact over a whole run and interleaved along it rather than run in blocks.
That second property matters: the unanswerable share and the follow-up share
are both taken by position, so a mix run in blocks would always perturb the
same kind of question.

A weight of `0`, or a name left out, is never written. That is the switch for
choosing what a run produces.

`GET /questions/plan` reads a topic's plan back before anything is written.

### Step 2 — The deal

**Where:** [`selection.py`](selection.py).

Within a topic the writer is offered a **sample** of facts.
`QUESTIONS_FACT_SAMPLE` caps how many, **divided between the passages the
sample holds**, so a wide sample offers both sides of what it is asking about.

The band the plan asked for decides the **shape of the sample**, and the shape
is what makes a band reachable at all — a question drawn from one passage
cannot be cross-document however it is phrased:

| Band | Shape | What the deal offers | Worth |
|---|---|---|---|
| `easy` | `single` | one passage | 0 |
| `medium` | `cross` | a passage in another document, ranked by shared vocabulary | 2 |
| `hard` | `bridge` | a bridging passage in another document — another file *and* another subject | 3 |

Passages are dealt **strided over the whole topic** rather than from its
start, and each is offered `QUESTIONS_SAMPLES_PER_PASSAGE` times. Ten
questions used to mean the first ten passages in document order, so two thirds
of a large topic was never asked about at all.

A topic sitting in one document has no cross-document question in it. The deal
falls back to the widest sample it can give — the nearest passage of the same
document, by ordinal — rather than writing nothing about that subject.

Likeness pairs the **passages**, and then it picks the **facts**. The
second passage's share of the sample is the facts closest to what the first
offered, not its own best: its own best is what it would have given a question
of its own, and two passages the corpus calls related still hold facts with
nothing between them. Offering those produced a question welding a date to a
category — *"Wodurch unterscheiden sich MT und modellbasiertes Testen
hinsichtlich Einordnung und erstmaliger Nennung?"* — which no honest question
spans. Measured over 120 real cross-document pairs, choosing the second side
for what it meets rather than for its own rank halved the pairs offered with
nothing in common, from 73 to 38.

**Cosine where the corpus is embedded, shared lemmas where it is not.** Both
measures are the same shape — a share in [0, 1], one over `passages.embedding`
and one over the lemma arrays — and the vectors are preferred because a lemma
overlap cannot see a synonym. Two passages about one subject in different
words share a direction and no vocabulary; Jaccard scores that pair 0, so the
sampler passed over the genuinely related pair and took an unrelated one
instead. `Testfall` against `Prüffall` and `Fehlerzustand` against `Defekt`
are both in this corpus.

A fact is weighed against the **nearest** of the facts already offered rather
than against their mean. A mean of several vectors points somewhere none of
them is, so a third fact would be measured against a subject the offer does
not hold; what the writer needs is a fact meeting one of the others, because
that is what a question spans.

The fallback is not a transition. A corpus extracted before the embedding
columns existed carries no vectors at all, and `make extract-embed` is what
gives it some.

Where nothing in the second passage meets the first, the sample is still
offered and the writer is told to narrow: the span rules end by saying that if
the facts have no single honest question between them, ask about one of them
and cite only that. `difficulty` is read off what it cited, so a question that
took the narrower option bands as what it actually is.

### Step 3 — Write one question

**Where:** [`generation.py`](generation.py), [`types.py`](types.py).

One shared rule block holds what is true of every question — do not name the
source, name the subject, one question, the language of the facts — and each
kind adds what it asks for, what its answer looks like, and one worked
example. Two prompts for one rule is how the two come to disagree.

Which facts a question actually cites is the **writer's** answer, not the
sample's. That distinction was missing at first and produced a measurable lie:
facts paired only because they came from different documents had nothing to do
with each other, the writer answered one and ignored the rest as its prompt
told it to, and the row was stored with a `cross_document` label earned by a
fact the question never used — 91 of the first 140 rows. A sample wider than
one passage now also carries an instruction to use both halves.

### Step 4 — Gate it

**Where:** [`checker.py`](checker.py), [`gates.py`](gates.py),
[`verifier.py`](verifier.py). See [The gates](#the-gates).

### Step 5 — Draw a balanced release

**Where:** [`balance.py`](balance.py). See
[The balanced release](#the-balanced-release).

## The thirteen kinds

Every one of them is a question **form**, never a subject, so the same list
applies to a manual, a contract, a policy or a report. Nothing in any prompt
names a domain.

`Level` is `cognitive_level`, declared by the kind and stored on every
question it writes — see [the other axis](#what-a-question-asks-of-a-reader)
for why it is not the band.

| Kind | Asks for | Answer | Passages | Level |
|---|---|---|---|---|
| `factoid` | one checkable value — how many, how much, by when | value | 1 | recall |
| `definition` | what a named thing or status is, as the material defines it | explanation | 1 | understand |
| `entity` | who does, decides, owns or must be told something | value | 1 | recall |
| `enumeration` | which things belong to a named set | list | 1 | understand |
| `condition` | when, or under what circumstances, something applies | list | 1 | apply |
| `reason` | why something is required, done, or the way it is | explanation | 1 | analyse |
| `procedure` | how something is done, or in what order | explanation | 1 | apply |
| `consequence` | what happens when something is or is not done | explanation | 1 | analyse |
| `comparison` | how two named things differ | list | 2 | analyse |
| `aggregation` | a total no single fact states on its own | value | 2 | analyse |
| `temporal` | what changed between two periods | list | 2 | analyse |
| `implication` | what must be true when two stated things both hold | explanation | 2 | analyse |
| `application` | which stated rule governs a case the material omits | explanation | 1 | apply |

### Which bands a type may be planned at

A band asks for the **shape of a sample**, so a type needing two passages
cannot be `easy` — `easy` offers one. That is the default, and for four of
the five two-passage types it is the only rule needed: a comparison drawn
from one passage is a question about one thing.

`application` is the exception, and it is why a type may also declare a
**floor** of its own. It needs one passage, because the rule it applies sits
in one — but it is still not a lookup, because the case it puts that rule to
is not in the material. Read only through the passage count it was planned
`easy`, which is the wrong shape for it.

So a type declares its floor and the passage count supplies the default. For
`application` that costs nothing and buys it the third of its slots it should
never have had: over ten slots, medium 8 and hard 2 where it was easy 4,
medium 4, hard 2.

It is the same distinction `cognitive_level` draws against `difficulty`,
applied a second time — how far an answer is spread is not what has to be
done with it once found.

### The answer form, and why it is a column

`answer_form` is `value`, `list` or `explanation`, declared by the kind. Every
gate that reads an answer reads it against that form.

This is the column the old set did not have, and not having it is why every
question in it was a lookup. One prompt demanded *"a short noun phrase, a few
words at most … never anything with a verb in it"*, and one structural gate
enforced it on every answer. Measured against six realistic answers, five were
refused as malformed:

```text
'weil die Risiken im Bankensektor gestiegen sind'                    -> malformed
'because risks in the banking sector increased'                      -> malformed
'by notifying the authority within four hours through the portal'    -> malformed
'submit the application, provide the business plan, and pay the fee' -> malformed
'EUR 15,000'                                                         -> accepted
```

Why, how, what-happens-if and which-things were not badly written. They were
**unwritable**. The verb rule now applies to a `value` alone; an `explanation`
is refused for carrying *no* verb, which is the opposite failure; and each
form has its own length bounds in `QUESTIONS_ANSWER_CHARS`.

### What a question asks of a reader

`difficulty` says how far an answer is spread — over two passages, two
documents, two subjects — which is how hard it is to **find**. It says nothing
about what has to be done once it is found, and a question spanning two
documents can still be a bare lookup with both of them in hand.

`cognitive_level` is the other axis, and it is derived from the question's type
the way difficulty is derived from its scopes. Nobody judges a row.

| level | types |
|---|---|
| `recall` | factoid, entity |
| `understand` | definition, enumeration |
| `apply` | condition, procedure, **application** |
| `analyse` | reason, consequence, comparison, aggregation, temporal, **implication** |

The two in bold are what make the column a measurement rather than a label.
Every other type's answer is *stated* in the passages — the recoverability
gate refuses one whose answer is not — so the level of a retrieval question
describes the shape of a lookup. These two are **derived**: the premises are in
the material and the conclusion is not, so answering means reasoning.

An `implication` puts two statements together and asks what they come to. An
`application` puts a rule the material gives to a case it does not mention.
Both need the conclusion to be absent — if the material already says it, that
is a `consequence`, and the gate refuses it.

`GET /questions` filters on the level and `GET /questions/quality` reports the
spread, which is the number that says whether a set is a retrieval benchmark
or a reasoning one: a set that is all `recall` is a lookup benchmark however
many of its questions reach two documents.

The column is NULL on every question written before it existed, and
**deliberately not backfilled** from `question_type`: a kind's level can
change, and a column filled in afterwards would say a row was judged when
nothing had judged it.

Because their answers are absent by construction, recoverability asks them the
wrong question, so they face one of their own — the same bargain `aggregation`
already had. Which one depends on how the type derives: `arithmetic` asks
whether the figures come to the total, `entailment` asks whether the conclusion
follows from the premises. They are not interchangeable: *do the arithmetic* is
the wrong instruction for a conclusion drawn from two rules, and *does this
follow* is the wrong one for a total, which follows from anything if the reader
is generous about addition.

### The three criteria, and the band they feed

Each is read off the facts the question reported citing. Each says something
different about what a chatbot has to do, so they are three columns rather than
one:

| | |
|---|---|
| `passage_scope` | `single_passage` or `multi_passage` — how many passages hold the answer |
| `document_scope` | `single_document` or `cross_document` — the one a retriever cannot fake: no single chunk holds the answer |
| `topic_scope` | `single_topic` or `multi_topic` — whether the question bridges two subjects |

`difficulty` is then **derived** rather than judged. Five things each count one
point — the three scopes above, an answer past `QUESTIONS_LONG_ANSWER_CHARS`,
and following another question — and the band is the total: 0–1 `easy`, 2
`medium`, 3+ `hard`.

Three and not four, because `cross_document` implies `multi_passage`: two
documents are two passages, so the three scopes total at most three. A
threshold of four would have made a question spanning two documents and two
subjects — the hardest thing a retriever faces — only medium. Nothing is
weighted, because a weighting is an opinion and the point of deriving
difficulty rather than judging it is that nobody has to hold one.

The band the plan asked for is stored as `planned_difficulty` beside the
`difficulty` the question turned out to be. The two disagree when the writer
cited fewer facts than it was offered, and the share that agree is in the
Questions page's Analysis fold: it is a measurement of the plan, not a fault in
the row.

## The gates

Applied cheapest first, because each one that fires saves the cost of those
behind it. Thirteen codes, and `questions.rejected_reason` holds exactly these.

| Gate | Rejects a question that | Costs |
|---|---|---|
| `malformed` | is not a question, carries no target answer when it claims one, is in the wrong language, or is one of its own facts handed back | nothing |
| `answer_too_short` | is scored against an answer below its form's floor | nothing |
| `answer_too_long` | answers past its form's ceiling — a value answered with a paragraph | nothing |
| `wrong_form` | answers in the wrong shape: a value describing an action, an explanation explaining nothing | nothing |
| `leaks_source` | names the material its answer is in: a quoted title, a numbered division, a bracketed reference, an author with a year, or what the residue call finds | nothing, or one question-only call |
| `off_topic` | was written to have no answer and is about nothing the material mentions, which any chatbot declines | nothing |
| `duplicate` | is a near twin of one already accepted | one index probe |
| `answerable_after_all` | was written to have no answer and turns out to have one | the same probe, or the round trip |
| `compound` | asks two things, so half an answer is neither right nor wrong | nothing |
| `unanchored` | nobody could have asked without the passage in front of them: it names too little, or it points at something only the passage holds | nothing, or one question-only call |
| `not_recoverable` | cites evidence its own answer is not in | the round trip |
| `answerable_elsewhere` | was written to have no answer and a passage it does not cite answers it | a lemma probe and a second call |
| `source_changed` | rests on a fact that no longer passes its own checks | nothing |

`source_changed` is the one no run of the writer produces. It is
`make questions-reverify` carrying a change made further up the pipeline
through to the questions resting on it.

### `compound`, and the gate that was deleted

`compound` is read off the tagger: **a question using more than one
interrogative word is two questions.** A finite-verb count does not separate
them — `Welche Arten von Kryptowerten gelten als reguliert?` carries two verbs
and is one question — and measured over 17 real questions the interrogatives
separated 16.

It replaced `wrong_type`, which is gone, and the reason is worth recording.
The verifier was asked whether a question was the kind it had been planned as,
and answered `true` for a bare `Wie viele Anlassprüfungen wurden 2025
durchgeführt?` written into a `reason` slot. It fired **zero times over 71
questions** while the kind was plainly wrong on six of the 27 accepted. A
yes-or-no judgement from a model this size is worth nothing, which is what the
phrasing judgement had already shown. What survives is structural: a kind
declares an answer form, and `wrong_form` reads the answer against it.

### There is no gate for "the question is its own fact rearranged"

That is a finding rather than an omission. One was written, in two
formulations, and measured against 61 real rows: both refused questions like
*"Wie hoch war die Arbeitslosenquote im August 2025?"* → `6,4 Prozent`, which
is as good as a benchmark question gets. For a single atomic fact, a good
question *is* the fact minus its answer — that is what asking about a fact
means. What separates a good one from a bad one is whether the answer is
determinate, and that is not lexical either. `not_recoverable` already carries
the judgement where it can be made.

### The round trip

`not_recoverable` is the one no similarity measure makes. A second model is
shown **only the cited passages** and asked to answer; the question survives if
what comes back carries every number, name and date the target answer asserts,
and enough of what it is about. How much is enough depends on the form: a
`value` is compared whole, because every word of one is the answer, while a
`list` or an `explanation` is compared by overlap — `QUESTIONS_ANSWER_OVERLAP`
— because demanding that every lemma of prose survive a paraphrase refuses
answers the verifier plainly found. Numbers are always exact whatever the form,
so `4 hours` never passes for `48 hours`.

`unanchored` and `leaks_source` ride on that same call, for nothing extra. Each
is a judgement rather than a measurement, and no structural check makes either
— but a model already looking at the question and the material can.

`unanchored` covers two ways of failing to stand alone, and the second has its
own measurement. A question may name too little — that is `anchored()`, which
counts what it names. Or it may name plenty and still **point outward**:
*"Wie groß ist der Unterschied … zwischen diesen beiden Lehrplänen?"* names
four things and is unanswerable, because nothing says which two. That one is
read off `PronType=Dem`, which every Universal Dependencies tagset marks, so no
word list per language is needed. It over-fires on purpose — half the questions
carrying a pointer set a case up first and refer back to it, which is the
`application` type working — so like `anchored()` it may only **veto** the
verifier's verdict, never make one.

**Measured over a full run of 3,131 questions, this gate fired zero times**, and
both of its halves are why. The verifier answered `self_contained: true` for
*"…zwischen den beiden Lehrplänen"* and for *"Kursmaterial, das diesem Lehrplan
entspricht"*, so the verdict never came. And the measurement would have missed
the first of those anyway: `diesen beiden` is `PronType=Dem`, but `den beiden`
is an article plus `PronType=Ind` and reads as pointing nowhere. Requiring two
permissive judgements to agree is what makes a gate inert.

What did move was the prompt. The `READS` rule against pointing at what the
asker cannot see took dangling references in accepted root questions from 9.4%
to 2.3% on its own. Treat this gate as unproven until the verdict is asked for
in a call of its own — see the note on that call's load below.

They are also the only gates here that are **opinions**, and an opinion needs
an independent holder. With `QUESTIONS_VERIFIER_MODEL` unset the writer marks
its own work, and one measured run rejected *"According to the ECB and NCAs,
who can conduct the due diligence check for an outsourcing arrangement?"* for
naming nothing — three of four rejections in that topic were false positives.
So the factory turns those gates off when no second model is named, and logs
the verdict instead. Recoverability stays on regardless, because that one is
checkable against the passage rather than a matter of taste.

### What that one call was carrying, and where those judgements went

`read()` used to ask for five things at once: whether the answer is in the
passages, what it is, whether the question names its source, what its subject
is, and whether it stands alone. The prompt had to warn that two of those were
"easy to confuse, so read both" — which is the shape of a call doing too much.

The run said so. `leaks_source` fired 9 times in 3,131 questions and
`self_contained` never fired at all, while `not_recoverable` — the judgement
the call exists for — fired 147 times. Then the harness measured it directly,
and the split by language is what settled it:

| | Both phrasing judgements right |
|---|---|
| English | 7/7 — 100% |
| German | 6/12 — **50%** |

Every miss was German, in a call whose recall half is good in both languages.
A judgement sharing a call with a harder task is answered in the language the
harder task is thinking in — and this corpus is German.

So `read()` now asks one thing, and the three phrasing judgements are answered
where each of them belongs:

| Judgement | Answered by | Measured |
|---|---|---|
| `subject` | `gates.subject` — spaCy noun chunks | **19/19**, against 16/19 for the model |
| `names_its_source` | `gates.cites_source`, a model for the residue | rules settle 5 cases with **0 errors**, including all four the model got wrong |
| `self_contained` | `phrasing.py`, asked only where a pointer was found | see below |

`cites_source` matches what a pattern can settle — a numbered division
(`Abschnitt 2.2`, `Kapitel 5`), a bracketed reference (`[R22]`), an author
with a year (`Beck 2003`), a named and numbered document — and returns
**None, never False**, because the absence of a pattern is not evidence.
`Welche Reviewverfahren beschreibt die Norm ISO/IEC 20246?` names a source and
`Warum wird ISO/IEC/IEEE 29119-4 erwähnt?` does not; they have the same shape
and differ by which of the standard and the answer the question is about. That
is a reading, so the residue goes to a model.

Two consequences worth knowing:

- **The phrasing gates now run before the round trip**, because none of them
  reads a passage. A question naming its own source is refused without ever
  paying for the call that would have answered it.
- **`QUESTIONS_PHRASING_MODEL`** is the dial. Unset it falls back to
  `QUESTIONS_VERIFIER_MODEL`, because what it judges is still the writer's
  work. It never sees a passage, so it can be far smaller than the verifier.

### `self_contained`, and why it fired zero times

Two permissive judgements had to agree, and **both halves were broken**.

The verifier answered `self_contained: true` for *"…zwischen den beiden
Lehrplänen"*. And the measurement would have missed it anyway: `den beiden` is
an article plus `PronType=Ind`, where only `diesen beiden` is `PronType=Dem`.
So the gate could not fire even when the verdict came.

The measurement now reads three things rather than one, and the middle one is
the case above:

| Reading | Catches | Why it is needed |
|---|---|---|
| `PronType=Dem` | `diesen beiden`, `this version` | every UD tagset marks it |
| a noun made **definite and counted** | `den beiden Lehrplänen`, `the two editions` | German tags `den` an article and `beiden` `PronType=Ind`; English tags `the two` article plus `NumType=Card`. Neither is a demonstrative, and both name a set of a known size the question never introduced |
| the anaphoric adjectives | `aforementioned`, `latter`, `besagt` | these carry no feature at all, so a short lemma list is the only thing available |

It still **over-fires on purpose** — `die drei Wege` is caught and is a
perfectly good question — so it may only ever offer a pointer, and the verdict
is the model's. What it may no longer do is stay silent about the phrasing the
gate was written for.

Three things about the round trip are load-bearing. The verifier is a
**different** model, because a model marking its own work recovers what it just
wrote and the gate then passes everything. The escape hatch is explicit — the
verifier answers whether the passage states it at all, not just what it says —
or the model confabulates rather than declining. And it sees only the cited
passages, never the corpus: what is being measured is the dataset, not a
retriever.

It catches what nothing else did. *"Ein Liquiditätsmanagementtool ist eine
einjährige Rückgabefrist"* survived an NLI model, an LLM judge and a structural
check, because it reads exactly like its passage. It does not survive being
asked, because recoverability is not similarity — and a paraphrase, a
decomposition and a resolved pronoun all survive it, which a similarity
threshold does not let them do.

## Unanswerable questions

Some questions are written to have **no answer in the corpus**, by perturbing a
verified fact just out of reach. These test whether a chatbot says it does not
know instead of inventing something, which is half of what this dataset is for.

`QUESTIONS_UNANSWERABLE_SHARE` sets how many are **attempted**, spread by
position rather than drawn at random, so a share of 0.10 is exactly one in ten
and is the same one in ten on a re-run. An unanswerable question is always
planned `easy` and from one passage: it is written by moving one fact out of
reach, so a second passage has nothing to do with it.

Attempted, not held: they now face two gates of their own, and those gates
bite. Over 3,131 questions, 0.10 produced 239 attempts and 101 accepted — 4.9%
of the accepted set, because 93 went as `answerable_after_all`. Set this above
the share the release is meant to hold and read the outcome off
`make questions-balance`.

Three gates exist only for these, and each closes a different hole:

| | |
|---|---|
| `answerable_after_all` | the corpus answers it after all, found by the dedup probe or the round trip |
| `answerable_elsewhere` | a passage the question does **not** cite answers it — `QUESTIONS_ELSEWHERE_PASSAGES` nearest by lemma, then one verifier call |
| `off_topic` | it is about nothing the material mentions at all, below `QUESTIONS_OFF_TOPIC_OVERLAP`, which any chatbot declines for the wrong reason |

The last one matters more than it looks. An unanswerable question that is also
off-topic is not a test: a chatbot refusing *"What is the capital of Mars?"*
has demonstrated nothing about whether it knows the limits of **this** corpus.

## Follow-up threads

A share of accepted questions get a **thread**: the question somebody would ask
next, up to `QUESTIONS_MAX_FOLLOWUPS` deep. `follows_id` and `thread_position`
carry it.

Each turn takes the next kind in `QUESTIONS_FOLLOWUP_TYPES`, cycled, so a
conversation moves from a value to the circumstances it applies in to the
reason behind it rather than asking the same kind of thing three times.

The cycle **starts where the thread sits** among the followed ones rather than
at the first kind, so a list longer than `QUESTIONS_MAX_FOLLOWUPS` is a
rotation and not a prefix with a dead tail. Starting every thread at the first
kind left `comparison` unwritten across 3,119 questions.

A follow-up **may lean on the conversation** — *"And for an urgent one?"* — and
that is the point: a chatbot answering one has to carry the thread, which is a
real capability and one no single-turn question tests. Two consequences follow
from it. The phrasing gate is **not** applied to a follow-up, because not
standing alone is what it is for; and the verifier is shown the conversation
when it judges recoverability, because read alone *"And for an urgent one?"*
has no answer in any passage.

Only an accepted, **answerable** root is followed. A thread whose first turn has
no answer has nothing to follow on from — the chatbot was supposed to say it did
not know — and one whose root a gate refused would be a conversation starting
with a question nobody would ask. A thread stops at the first follow-up a gate
refuses: the refused one is stored as drop-rate evidence like any other, but a
third turn after a discarded second is a conversation with a hole in it.

Each follow-up is another writer call and another verifier call, so a thread
multiplies what a topic costs.

## Coverage, and what the topic count cannot say

`topics_covered` reads 38 of 38 whenever every topic produced one accepted
question, which is a low bar: a topic holds eighty passages. So
`/questions/quality` also reports how much of the MATERIAL the accepted
questions reach.

| | Measured over this corpus |
|---|---|
| passages | 1,495 |
| passages a validated fact rests on | 1,167 |
| **passages an accepted question rests on** | **880 — 75% of askable, 59% of all** |
| validated facts | 6,242 |
| **facts an accepted question cites** | **2,114 — 34%** |
| questions per passage asked about | 1.98 |
| questions per fact cited | 0.82 |

Two things about the denominators.

**Passages are counted against those a validated fact rests on**, not against
every passage. A passage no fact rests on cannot be asked about, so counting
it here would report extraction's refusals as this stage's gap. The 328 that
are missing from 1,495 are passages the extractor skipped or whose every
fact a check refused.

**Questions per fact below 1 is ordinary.** A question may cite several
facts and each one counts as asked about, so the ratio falls as questions
get wider rather than as they get fewer.

The figure to read for headroom is the fact one. At 34%, two thirds of the
validated corpus has never been asked about — and because selection skips
the facts an accepted question already rests on, another run reaches that
two thirds rather than re-asking the first third.

## The balanced release

Accepting a question says it is sound. It says nothing about what the SET looks
like, and the two are different problems: every gate can do its job and still
leave a set that is 45% unanswerable and 96.5% easy, because what survives a
filter is whatever the material happened to offer. That is what one measured
run came out as.

So the run overgenerates and the composition is chosen afterwards.
`make questions-balance` fills a quota out of everything accepted and writes
`questions.release_id` on what it drew. Three shares are held at once:

| | |
|---|---|
| `QUESTIONS_RELEASE_UNANSWERABLE` | The most of the release that may be questions with no answer. A ceiling, not a target. |
| `QUESTIONS_RELEASE_DIFFICULTY` | How it spreads over the bands, over the whole set rather than over the answerable part of it. |
| `QUESTIONS_TYPE_MIX` | Reused, so a type a deployment turned off gets no quota it cannot fill. |

They are **marginals, not a joint distribution**, and that is deliberate. An
even spread of kinds *within* each band is not reachable: a `factoid` answers
with a value, a value is short, and a short answer cannot earn the length point
a `hard` question generally needs. Asking for both marginals is achievable;
asking for their product is asking the corpus to be something it is not.

`QUESTIONS_RELEASE_SIZE=0` draws the largest release the pool can fill without
missing a quota, which is usually what is wanted: the size a balanced set can
reach is fixed by whichever bucket is furthest from supplying its share, and
that is a property of the run rather than a number to pick. The command reports
what it drew, the yield — how much of the accepted pool made it in — and names
any quota the pool came up short on.

Nothing is deleted and nothing is rewritten. A release is a column on the rows
already there, so the questions left out stay queryable and available to the
next draw, and running it again with different shares replaces the first.

The number to read is that yield, not the share of rows that were accepted.
Every draft the writer produced is stored, refused ones included — including
the first attempt where `QUESTIONS_RETRIES` asked for a second — because they
are the drop-rate evidence, and hiding them would make the writer look better
than it is.

## The embedding model

`EMBEDDING_MODEL` is the one pretrained model in this pipeline, and the one
thing in it that does **not** move when you change providers. It runs locally,
in the worker, whatever `LLM_MODEL` names — a hosted provider serves the writer
and the verifier and has nothing to do with this.

It does two jobs, and they are one setting because they must agree:

| | |
|---|---|
| Chunking | Counts a passage's tokens. `EMBEDDING_MAX_TOKENS` is the model's window and so the longest passage the chunker emits. |
| The `duplicate` gate | Embeds each question into `questions.embedding`, which the HNSW index serves. `answerable_after_all` reads the same probe. |

A passage sized by one tokenizer and embedded by another is silently truncated
at embed time, and a benchmark built on truncated passages misreports its own
coverage. One name is what makes that impossible.

**Changing it is a migration, not a setting.** `questions.embedding` is
`vector(1024)`, and the worker refuses a model of another width at start-up
rather than letting the database refuse it one row at a time with nothing
saying why. A different model means altering that column, re-embedding every
stored question, and re-chunking if its window differs — so the passages are
sized in the unit they are embedded in. The default,
`intfloat/multilingual-e5-large`, is 1024 wide with a 512-token window and
covers German and English.

It stays local deliberately. The gate runs once per candidate question, which
is tens of thousands of vectors over a corpus, and a hosted embedding API would
bill for every one of them to answer a question — "have I already asked this?"
— that a 560M-parameter model answers well. It is also what keeps a question
measured in the same space the corpus was put in.

The cost is the first start: 2.2 GB of weights into the `models` volume before
`question-worker` claims anything, which is why the topics sit `pending` and
`make logs` is where it says so. Later starts read the volume. `question-worker`
is given 6 GB because of it, and that is the number to remember when deciding
between more worker containers and more lanes inside one — the weights are per
container, not per lane.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **litellm** + **instructor** | [`generation.py`](generation.py), [`verifier.py`](verifier.py) | One model id names the provider; instructor is what makes the answer a typed object rather than prose to parse |
| **sentence-transformers** | [`embedding.py`](embedding.py) | `EMBEDDING_MODEL` locally, for the dedup probe. No hosted embedding API bills for tens of thousands of "have I asked this?" |
| **pgvector** | `questions.embedding` | An HNSW index over 1024-wide vectors, so the dedup gate is one probe rather than a scan |
| **spaCy** | [`gates.py`](gates.py) | Interrogatives for `compound`, finite verbs for `wrong_form`, lemmas for the overlap measures. No model call |
| **SQLAlchemy** | [`queue.py`](queue.py), [`catalog.py`](catalog.py) | The topic queue, and reading the questions back |

## Inputs and outputs

**In:** one topic with `question_status = 'pending'`, the facts of the passages
it is strongest in, and those passages.

**Written, in PostgreSQL:**

| Table | Holds |
|---|---|
| `questions` | the question, its target answer, its type, form and cognitive level, the three scopes and the band derived from them, the planned band beside it, the thread columns, the embedding, the release id, and the gate that stopped it when one did |
| `question_facts` | which facts each question cites. A trigger deletes a question once its last citation is gone |

**Serves:** `/questions`, `/questions/{id}`, `/questions/plan`,
`/questions/quality`, `PATCH /questions/{id}`, and the five queue verbs under
`/questions`. See the [API README](../api/README.md).

**Command line:**

```bash
make questions-start            # queue every topic not yet asked about
make questions                  # drain the queue here, in the foreground
make questions-status
make questions-stop
make questions-retry
make questions-rerun
make questions-start TOPIC=7    # any of the five, over one topic
make questions-reverify         # re-check stored questions; no model is called
make questions-balance          # draw the balanced release
```

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `QUESTIONS_PER_TOPIC` | 60 | How many questions to aim for per topic, and so how many of its passages are asked about. This times the topic count is what a full run costs. Not below the number of kinds with a weight, or a topic never sees some of them |
| `QUESTIONS_FACT_KINDS` | `atomic,summary,outline,bridge` | Which kinds of fact a question may be written from |
| `QUESTIONS_FACT_SAMPLE` | 3 | How many of a topic's facts are offered per call, divided between the passages the sample holds |
| `QUESTIONS_SAMPLES_PER_PASSAGE` | 3 | How many times one passage is dealt before the stride moves on |
| `QUESTIONS_TYPE_MIX` | thirteen kinds, weight 1 each | Which kinds are written and in what proportion, as `kind:weight`. A weight of 0, or a name left out, is never written |
| `QUESTIONS_DIFFICULTY_MIX` | `easy:2,medium:3,hard:3` | Which bands the plan aims for. A request for a shape of sample; the band itself stays derived |
| `QUESTIONS_ANSWER_CHARS` | `value:1:80,list:3:300,explanation:20:600` | The shortest and longest target answer per form, as `form:min:max` |
| `QUESTIONS_ANSWER_OVERLAP` | 0.6 | How much of a list or an explanation has to come back for the verifier to have recovered it. Numbers are always exact |
| `QUESTIONS_UNANSWERABLE_SHARE` | 0.10 | What share of questions are written to have no answer in the corpus |
| `QUESTIONS_OFF_TOPIC_OVERLAP` | 0.3 | Below this share of shared lemmas, an unanswerable question is about nothing the material covers |
| `QUESTIONS_ELSEWHERE_PASSAGES` | 4 | How many uncited passages the `answerable_elsewhere` probe reads |
| `QUESTIONS_FOLLOWUP_SHARE` | 0.5 | What share of accepted questions get a follow-up thread |
| `QUESTIONS_MAX_FOLLOWUPS` | 2 | How far a thread may run past its root |
| `QUESTIONS_FOLLOWUP_TYPES` | `condition,reason,comparison` | The kinds the turns of a thread take, cycled |
| `QUESTIONS_RETRIES` | 1 | How many further attempts a refused candidate gets. Every attempt is stored |
| `QUESTIONS_LONG_ANSWER_CHARS` | 60 | Where an answer starts counting towards the difficulty band. Not a gate |
| `QUESTIONS_DUPLICATE_COSINE` | 0.93 | How alike two questions must be before the later one is thrown away |
| `QUESTIONS_RELEASE_SIZE` | 0 | How many to draw. `0` is the largest the pool can fill without missing a quota |
| `QUESTIONS_RELEASE_UNANSWERABLE` | 0.10 | The most of a release that may have no answer. A ceiling |
| `QUESTIONS_RELEASE_DIFFICULTY` | `easy:1,medium:1,hard:1` | How a release spreads over the bands |
| `QUESTIONS_MODEL` | unset | A different writer from the rest of the pipeline. Unset means `LLM_MODEL` |
| `QUESTIONS_VERIFIER_MODEL` | unset | The second model. Naming the writer's own, or none, turns off the gates only an independent model may apply; the worker warns on every start |
| `QUESTIONS_PHRASING_MODEL` | unset | The model asked what a question's wording amounts to, where a rule has not settled it. Never sees a passage, so it can be far smaller than the verifier. Unset calls `QUESTIONS_VERIFIER_MODEL` |

The length bounds are per form and measured rather than guessed: a floor of 15
on values refused 41% of the answers this corpus had accepted — `70%`, `2025`
and `Bafin` among them, which are the most unambiguously scoreable answers
there are. One line in `backend.env` changes any of them, and the share each
refuses is in the Questions page's Analysis fold either way.

## Tests

```sh
poetry run pytest tests/unit/questions
poetry run pytest tests/integration/database/test_release.py
```

| File | Covers |
|---|---|
| [`test_planning.py`](../../tests/unit/questions/test_planning.py) | What a topic is planned to be asked, before anything is written |
| [`test_selection.py`](../../tests/unit/questions/test_selection.py) | What the deal offers the writer, and how wide it is |
| [`test_generation.py`](../../tests/unit/questions/test_generation.py) | What the writer asks the model, and what it makes of the answer |
| [`test_verification.py`](../../tests/unit/questions/test_verification.py) | Every gate, and the order they are applied in |
| [`test_competence.py`](../../tests/unit/questions/test_competence.py) | The types whose answers are reasoned to rather than found |
| [`test_balance.py`](../../tests/unit/questions/test_balance.py) | Choosing a balanced release out of everything a run accepted |
| [`test_models.py`](../../tests/unit/questions/test_models.py) | Which model writes a question, and which one checks it |
| [`test_question_service.py`](../../tests/unit/questions/test_question_service.py) | What the service does with a topic, short of touching a database |
| [`test_embedding.py`](../../tests/unit/questions/test_embedding.py) | The embedding behind the dedup gate, against the real model. Skips unless `EMBEDDING_MODEL` is already in the Hugging Face cache |
| [`test_question_queue.py`](../../tests/integration/database/test_question_queue.py) | Two queues on one table, against the database that has to keep them apart |
| [`test_release.py`](../../tests/integration/database/test_release.py) | Drawing a balanced release, against the table that stores it |
| [`test_trigger.py`](../../tests/integration/database/test_trigger.py) | The trigger that deletes a question once its last fact is gone |
| [`test_frontend_vocabularies.py`](../../tests/static/test_frontend_vocabularies.py) | That the Questions page lists exactly the gates the checker can reject under |

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A refit rewrites every topic but keeps every question.** Questions belong
  to their facts, not to a topic. A fit returns every topic to `new` and the
  questions are written again — but selection skips the facts an accepted
  question already rests on, so the second run writes only about what the
  first did not reach.
- **Re-extracting a single document is quieter and worse than re-extracting
  the corpus.** `question_facts` cascades from `facts` and a trigger deletes a
  question once its last citation is gone, so `extract-rerun` over the corpus
  takes the questions with it. A cross-document question that loses one of two
  citations is *not* deleted, and its stored difficulty stops being true.
  `questions-reverify` is what finds that.
- **`questions-reverify` only ever rejects.** Accepting is a person's
  decision, and a re-check that un-rejected would overturn one on its next
  run.
- **`planned_difficulty` disagreeing with `difficulty` is not a fault.** The
  writer cited fewer facts than it was offered. It is a measurement of the
  plan.
- **The first start of `question-worker` looks like nothing happening.** It
  downloads 2.2 GB of embedding weights before it claims anything. The topics
  sit `pending` and `make logs` is where it says so.
- **`QUESTIONS_PER_TOPIC` is what a run costs, twice over.** Each candidate is
  a writer call and a verifier call, and `QUESTIONS_FOLLOWUP_SHARE` adds a
  pair on top for every thread.
