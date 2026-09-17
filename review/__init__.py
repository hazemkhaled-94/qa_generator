"""Human review of what the models decided, through Argilla.

Three things in this pipeline are a model's judgement and have somewhere
for a person to disagree:

    a fact's verdict      facts.reviewed_verdict
    a topic's name        topics.label, topics.include_in_coverage
    a question's fate     questions.status

Those are the three datasets. Each already had a human-owned column or a
route of its own before this existed - `PATCH /questions/{id}` and
`PATCH /topics/{id}` are the same writes - so what this adds is a place to
make a hundred of those decisions in a row instead of one at a time
through a table.

The database decides. Argilla holds a copy of the rows put in front of
somebody and the answers they gave; `--pull` brings the answers home and
the copy is disposable. Delete the Argilla dataset and the pipeline has
lost nothing.

Runs on the host, not in a container. Nothing in the pipeline calls it.

Nothing is re-exported: a caller names the submodule it wants.
"""
