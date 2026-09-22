"""Build the labelled train/dev/test splits for both tasks into ./data as JSONL.

spam:  Enron-Spam (SetFit/enron_spam), real corporate mail with ham/spam labels.
route: Bitext customer support, 11 categories with clean, template-derived labels.
scenario_sv: MASSIVE (Swedish split, human-localised), 18 voice-assistant scenarios.

Every split is deterministic: rows are ordered by a hash of their text, not by a random seed, so
rebuilding gives the same test set. Texts that appear in more than one split are removed from
training, so spaCy is never scored on something it has memorised.

    uv run python data.py
"""

from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import duckdb

from labels import CATEGORIES, SCENARIOS
from records import Example, write_jsonl

ENRON = "https://huggingface.co/api/datasets/SetFit/enron_spam/parquet/default/{split}/0.parquet"
BITEXT = (
    "https://huggingface.co/api/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/parquet/default/train/0.parquet"
)
MASSIVE_SV = "https://huggingface.co/api/datasets/mteb/amazon_massive_scenario/parquet/sv/{split}/0.parquet"
OUT = Path(__file__).parent / "data"

# Every contestant sees the same text, cut at this length: Jev works best on focused state, and a
# 40 kB newsletter tells a classifier nothing its first 3 kB did not.
MAX_CHARS = 3000
SPAM_TEST_PER_CLASS = 150
SPAM_DEV_PER_CLASS = 500
SPAM_TRAIN_PER_CLASS = 6000
ROUTE_TEST_PER_CLASS = 25
ROUTE_DEV_PER_CLASS = 40
SCENARIO_TEST_PER_CLASS = 20
SCENARIO_DEV_PER_CLASS = 20


def spam_splits(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    con.sql(f"""
        create or replace temp table enron as
        select distinct on (text)
               split, left(trim(text), {MAX_CHARS}) as text, label_text as label, message_id::varchar as message_id
        from (select 'train' as split, * from '{ENRON.format(split="train")}'
              union all
              select 'test' as split, * from '{ENRON.format(split="test")}')
        where length(trim(text)) > 20
        order by text, split  -- a text in both splits keeps its 'test' row, so it never trains
    """)
    test = _stratified(con, "select * from enron where split = 'test'", "label", SPAM_TEST_PER_CLASS)
    rest = _stratified(
        con,
        "select * from enron where split = 'train'",
        "label",
        SPAM_DEV_PER_CLASS + SPAM_TRAIN_PER_CLASS,
    )
    dev, train = _split_head(rest, "label", SPAM_DEV_PER_CLASS)
    return {"test": test, "dev": dev, "train": train}


def route_splits(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    con.sql(f"""
        create or replace temp table bitext as
        select distinct on (lower(instruction))
               instruction as text, lower(category) as label, intent,
               md5(instruction) as message_id
        from '{BITEXT}'
        order by lower(instruction)
    """)
    unknown = con.sql(f"select distinct label from bitext where label not in {tuple(CATEGORIES)}").fetchall()
    assert not unknown, f"Bitext has categories labels.py does not know: {unknown}"

    everything = _stratified(con, "select * from bitext", "label", None)
    test, rest = _split_head(everything, "label", ROUTE_TEST_PER_CLASS)
    dev, train = _split_head(rest, "label", ROUTE_DEV_PER_CLASS)
    return {"test": test, "dev": dev, "train": train}


def scenario_sv_splits(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    """MASSIVE's own train/validation/test split, de-duplicated, with test and dev texts kept out of train."""
    con.sql(f"""
        create or replace temp table massive as
        select distinct on (lower(text)) split, text, label, id as message_id
        from (select 'test' as split, * from '{MASSIVE_SV.format(split="test")}'
              union all
              select 'train' as split, * from '{MASSIVE_SV.format(split="train")}'
              union all
              select 'validation' as split, * from '{MASSIVE_SV.format(split="validation")}')
        order by lower(text), split  -- test < train < validation: a shared text stays in test
    """)
    unknown = con.sql(f"select distinct label from massive where label not in {tuple(SCENARIOS)}").fetchall()
    assert not unknown, f"MASSIVE has scenarios labels.py does not know: {unknown}"
    return {
        "test": _stratified(con, "select * from massive where split = 'test'", "label", SCENARIO_TEST_PER_CLASS),
        "dev": _stratified(con, "select * from massive where split = 'validation'", "label", SCENARIO_DEV_PER_CLASS),
        "train": _stratified(con, "select * from massive where split = 'train'", "label", None),
    }


def _stratified(con: duckdb.DuckDBPyConnection, query: str, by: str, per_class: int | None) -> list[dict]:
    """Up to `per_class` rows of each `by` value, in a fixed hash order within each class."""
    limit = f"where rank <= {per_class}" if per_class else ""
    rel = con.sql(f"""
        select * exclude (rank) from (
            select *, row_number() over (partition by {by} order by hash(text)) as rank
            from ({query})
        ) {limit}
        order by {by}, hash(text)
    """)
    return [dict(zip(rel.columns, row, strict=True)) for row in rel.fetchall()]


def _split_head(rows: list[dict], by: str, head: int) -> tuple[list[dict], list[dict]]:
    """The first `head` rows of each class, and everything after them. `rows` must be grouped by `by`."""
    groups = [list(group) for _, group in groupby(rows, key=itemgetter(by))]
    return [r for g in groups for r in g[:head]], [r for g in groups for r in g[head:]]


def write(task: str, splits: dict[str, list[dict]]) -> None:
    for split, rows in splits.items():
        path = OUT / f"{task}_{split}.jsonl"
        examples = [Example.model_validate({k: v for k, v in r.items() if k in Example.model_fields}) for r in rows]
        write_jsonl(path, examples)
        counts = dict(sorted(Counter(e.label for e in examples).items()))
        print(f"data/{path.name}: {len(examples)} rows {counts}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    con = duckdb.connect()
    write("spam", spam_splits(con))
    write("route", route_splits(con))
    write("scenario_sv", scenario_sv_splits(con))


if __name__ == "__main__":
    main()
