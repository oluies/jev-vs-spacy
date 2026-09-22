"""Braintrust bake-off: spaCy vs Jev vs Claude on spam, support routing and Swedish routing.

One experiment per (task, contestant), all in one Braintrust project, so the UI can put any two
side by side on identical rows. A contestant whose API key is not set is skipped with a notice.

    uv run python eval_email.py                                   # log to Braintrust
    uv run braintrust eval --no-send-logs eval_email.py           # local only, prints the summaries
    LIMIT=20 uv run braintrust eval --no-send-logs eval_email.py  # smoke test

Log with `python`, not `braintrust eval`: the CLI ignores `base_experiment_name` (braintrust 0.41).

Latency per row is the span duration Braintrust records for the task; tokens and model-specific
probabilities are in each row's metadata.
"""

import sys
from itertools import groupby
from operator import attrgetter
from pathlib import Path

from braintrust import Eval, EvalCase

from contestants import contestants
from records import Example, read_jsonl
from scorers import ROUTE_SCORERS, SPAM_SCORERS
from settings import settings

TASKS = {
    "spam": (SPAM_SCORERS, "Enron-Spam test split: is this email spam?"),
    "route": (ROUTE_SCORERS, "Bitext support messages: which of 11 teams handles this?"),
    "scenario_sv": (ROUTE_SCORERS, "MASSIVE, Swedish: which of 18 voice-assistant skills handles this?"),
}


def cases(task: str) -> list[EvalCase[dict, str]]:
    """The test split as Braintrust cases, optionally cut to the first LIMIT rows of each class."""
    rows = read_jsonl(Path(__file__).parent / "data" / f"{task}_test.jsonl", Example)  # grouped by label
    kept = [r for _, group in groupby(rows, key=attrgetter("label")) for r in list(group)[: settings.limit or None]]
    return [
        EvalCase(
            input={"text": r.text},
            expected=r.label,
            metadata=r.model_dump(include={"message_id", "intent"}, exclude_none=True) | {"expected_label": r.label},
        )
        for r in kept
    ]


for contestant in contestants():
    if contestant.missing_key:
        print(f"skipping {contestant.name}: {contestant.needs} is not set", file=sys.stderr)
        continue
    for task, run in contestant.tasks.items():
        scorers, description = TASKS[task]
        Eval(
            settings.braintrust_project,
            experiment_name=f"{task}-{contestant.name}",
            description=description,
            data=cases(task),
            task=run,
            scores=scorers,
            metadata={"task": task, "contestant": contestant.name, "model": contestant.model},
            tags=[task, contestant.name],
            max_concurrency=settings.concurrency,
            # Diff every contestant against fully trained spaCy on the same task; without this
            # Braintrust diffs against whichever experiment happened to run last.
            base_experiment_name=None if contestant.name == "spacy" else f"{task}-spacy",
        )
