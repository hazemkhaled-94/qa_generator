"""The evaluation phase: an LLM judge over every artefact, never a gate.

Called `assessment` rather than `evaluation` because the root
`evaluation/` package is already the golden-set runner that scores a model
against `cases.py` on the host. Two packages of one name on two import
roots is a `make` target that picks whichever `sys.path` happened to put
first, so the phase is `assessment` everywhere the code says it and "the
evaluation phase" everywhere a person does.
"""
