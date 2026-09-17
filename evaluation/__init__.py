"""The golden cases, and scoring a served model against them in Phoenix.

`make test-eval` measures a model against these cases and prints the
numbers, which is the right shape for a pull request and the wrong shape
for a question like "is this prompt better than last week's". This is the
other half: the same cases as a Phoenix dataset, and a run as an
experiment beside it, so two are comparable.

    evaluation/cases.py        the cases, as data, read by both
    evaluation/experiments.py  the dataset and the scoring
    evaluation/run.py          the command line

Never a gate. It calls a real served model, so its numbers move between
versions, between quantisations and between two runs at the same
temperature.

Nothing is re-exported: a caller names the submodule it wants.
"""
