"""Braintrust scorers. All deterministic: the labels are known, so no LLM judge is needed.

Braintrust averages each named score over the rows that produced it and skips rows that return
`None`. That turns per-row scores into the usual classifier metrics without a custom summary:
"recall · x" is scored only on rows whose true label is x, "precision · spam" only on rows the
model called spam.
"""

from braintrust import Score

from labels import SPAM


def correct(output: dict, expected: str, **_) -> Score:
    return Score(name="accuracy", score=float(output["label"] == expected))


def recall_by_class(output: dict, expected: str, **_) -> Score:
    return Score(name=f"recall · {expected}", score=float(output["label"] == expected))


def spam_precision(output: dict, expected: str, **_) -> Score | None:
    """Of the mail the model would have quarantined, how much really was spam."""
    if output["label"] != SPAM:
        return None
    return Score(name="precision · spam", score=float(expected == SPAM))


SPAM_SCORERS = [correct, recall_by_class, spam_precision]
ROUTE_SCORERS = [correct, recall_by_class]
