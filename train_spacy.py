"""Train one spaCy text classifier per task from ./data, into ./models/<task>/model-best.

    uv run python data.py          # first, once
    uv run python train_spacy.py   # both tasks, a few minutes on a laptop CPU
    uv run python train_spacy.py spam
    uv run python train_spacy.py --shots 20   # only 20 labelled examples per class -> models/<task>-20shot

The few-shot models answer the question a zero-shot model like Jev raises: how much labelled data
does spaCy need before it wins?
"""

import argparse
from itertools import groupby
from operator import attrgetter
from pathlib import Path

import spacy
from spacy.cli.train import train
from spacy.tokens import DocBin

from labels import CATEGORIES, HAM, SCENARIOS, SPAM
from records import Example, read_jsonl

ROOT = Path(__file__).parent
LABELS = {"spam": [SPAM, HAM], "route": list(CATEGORIES), "scenario_sv": list(SCENARIOS)}
# Swedish gets spaCy's Swedish tokenizer and the pretrained sv_core_news_md vectors.
CONFIGS = {"scenario_sv": "textcat_sv.cfg"}
LANGS = {"scenario_sv": "sv"}


def read(task: str, split: str) -> list[Example]:
    return read_jsonl(ROOT / "data" / f"{task}_{split}.jsonl", Example)


def to_docbin(rows: list[Example], labels: list[str], lang: str) -> DocBin:
    """One-hot `doc.cats` over every label, which is what an exclusive `textcat` trains against."""
    nlp = spacy.blank(lang)
    docs = (nlp.make_doc(r.text) for r in rows)
    return DocBin(docs=[_labelled(doc, row.label, labels) for doc, row in zip(docs, rows)])


def _labelled(doc, label: str, labels: list[str]):
    doc.cats = {name: float(name == label) for name in labels}
    return doc


def first_per_class(rows: list[Example], shots: int | None) -> list[Example]:
    """The first `shots` rows of each label (files are grouped by label, in a fixed hash order)."""
    return [r for _, group in groupby(rows, key=attrgetter("label")) for r in list(group)[:shots]]


def model_name(task: str, shots: int | None) -> str:
    return task if shots is None else f"{task}-{shots}shot"


def train_task(task: str, shots: int | None) -> Path:
    return train_model(task, model_name(task, shots), first_per_class(read(task, "train"), shots))


def _small_data(rows: int) -> dict:
    """A small training set is only a few batches per epoch: give it more epochs, checked more often."""
    match rows:
        case n if n < 1000:
            return {"training.max_epochs": 60, "training.eval_frequency": 20}
        case n if n < 5000:
            return {"training.max_epochs": 30, "training.eval_frequency": 50}
        case _:
            return {}


def train_model(task: str, name: str, rows: list[Example]) -> Path:
    """Train `models/<name>` for `task` on `rows`, picking the best
    checkpoint on the task's gold-labelled dev split."""
    corpus = ROOT / "corpus" / name
    corpus.mkdir(parents=True, exist_ok=True)
    lang = LANGS.get(task, "en")
    to_docbin(rows, LABELS[task], lang).to_disk(corpus / "train.spacy")
    to_docbin(read(task, "dev"), LABELS[task], lang).to_disk(corpus / "dev.spacy")

    output = ROOT / "models" / name
    train(
        ROOT / "configs" / CONFIGS.get(task, "textcat.cfg"),
        output,
        overrides={"paths.train": str(corpus / "train.spacy"), "paths.dev": str(corpus / "dev.spacy")}
        | _small_data(len(rows)),
    )
    return output / "model-best"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", metavar="task", help=f"any of {list(LABELS)} (default: all)")
    parser.add_argument("--shots", type=int, help="train on only this many examples per class")
    args = parser.parse_args()
    if unknown := set(args.tasks) - set(LABELS):
        parser.error(f"unknown task(s) {sorted(unknown)}; choose from {list(LABELS)}")
    for task in args.tasks or list(LABELS):
        print(f"\n=== {model_name(task, args.shots)} ===")
        print(f"saved {train_task(task, args.shots)}")


if __name__ == "__main__":
    main()
