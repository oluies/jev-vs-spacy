"""Jev labels, spaCy learns: train a local spaCy model on labels Jev produced, not on human labels.

For a categorisation task this draws a random pool of POOL messages from the training split, as if
they were unlabelled mail, has Jev label them, and trains three spaCy models on that same pool:

    models/<task>-gold<N>   the dataset's own labels: the ceiling for a pool this size
    models/<task>-jev<N>    every label Jev gave
    models/<task>-jev80_<N> only the labels Jev gave with confidence >= 0.8

The gold labels are used only for the gold model and to report how accurate Jev's labels were.
Model selection uses the task's small gold dev split for every variant. That split stands in for
the hand-checked validation set you would keep in practice.

    TYPESAFE_API_KEY=... uv run python distill.py route scenario_sv --pool 2000
    uv run braintrust eval --no-send-logs eval_email.py   # the new models appear as contestants
"""

import argparse
import asyncio
import logging
import random
from collections import Counter
from pathlib import Path

from contestants import SCHEMAS, _agent_task, _jev
from records import Example, JevLabelled, read_jsonl, write_jsonl
from train_spacy import read, train_model

ROOT = Path(__file__).parent
CONFIDENT = 0.8


class _NoHooks:
    def meta(self, **_):
        pass


def sample_pool(task: str, size: int, seed: int = 0) -> list[Example]:
    """A uniform random sample of the training split: no use of the labels, unlike a stratified one."""
    rows = read(task, "train")
    return random.Random(seed).sample(rows, min(size, len(rows)))


async def jev_label(task: str, pool: list[Example], concurrency: int = 8) -> list[JevLabelled]:
    schema, field, to_label = SCHEMAS[task]
    run = _agent_task(lambda: _jev(schema), field, to_label)
    gate = asyncio.Semaphore(concurrency)

    async def one(row: Example) -> JevLabelled:
        async with gate:
            out = await run({"text": row.text}, _NoHooks())
        return JevLabelled(
            message_id=row.message_id, text=row.text, gold=row.label, label=out["label"], confidence=out["confidence"]
        )

    return await asyncio.gather(*map(one, pool))


def _confident(row: JevLabelled) -> bool:
    return row.confidence is not None and row.confidence >= CONFIDENT


def variants(labelled: list[JevLabelled], size: int) -> dict[str, list[Example]]:
    """Training rows per model suffix, all from the same pool."""
    return {
        f"gold{size}": [r.model_copy(update={"label": r.gold}) for r in labelled],
        f"jev{size}": labelled,
        f"jev80_{size}": [r for r in labelled if _confident(r)],
    }


def report(task: str, labelled: list[JevLabelled]) -> None:
    confident = [r for r in labelled if _confident(r)]
    accuracy = lambda rows: sum(r.label == r.gold for r in rows) / len(rows)  # noqa: E731
    print(
        f"{task}: Jev labelled {len(labelled)} messages, {accuracy(labelled):.1%} agree with gold; "
        f"{len(confident)} ({len(confident) / len(labelled):.0%}) at confidence >= {CONFIDENT}, "
        f"{accuracy(confident):.1%} agree"
    )
    missing = {r.gold for r in labelled} - {r.label for r in confident}
    if missing:
        print(f"  classes with no confident Jev label (the jev80 model never sees them): {sorted(missing)}")
    print("  most common disagreements:", Counter(f"{r.gold}->{r.label}" for r in labelled if r.label != r.gold).most_common(5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="+", choices=[t for t in SCHEMAS if t != "spam"])
    parser.add_argument("--pool", type=int, default=2000)
    args = parser.parse_args()
    logging.disable(logging.INFO)

    for task in args.tasks:
        cache = ROOT / "data" / f"{task}_pool{args.pool}_jev.jsonl"
        if cache.exists():  # labelling costs API calls: reuse a previous run's labels
            labelled = read_jsonl(cache, JevLabelled)
        else:
            labelled = asyncio.run(jev_label(task, sample_pool(task, args.pool)))
            write_jsonl(cache, labelled)
        report(task, labelled)
        for suffix, rows in variants(labelled, args.pool).items():
            print(f"\n=== {task}-{suffix}: {len(rows)} rows ===")
            print(f"saved {train_model(task, f'{task}-{suffix}', rows)}")


if __name__ == "__main__":
    main()
