"""The shapes of the JSONL files under data/, validated when read and written.

Every file the scripts pass to each other goes through these models, so a missing field, a
wrong type or a stray key fails at the file boundary rather than inside training or an eval.
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Example(BaseModel):
    """One labelled text: a row of `data/<task>_<split>.jsonl`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str
    text: str
    label: str
    intent: str | None = None  # Bitext's finer-grained intent, kept for error analysis


class JevLabelled(Example):
    """A pool row labelled by Jev (`label`), with the dataset's own label kept as `gold`."""

    gold: str
    confidence: float | None


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    # Split on "\n" only: model_dump_json leaves U+2028 and friends unescaped inside strings,
    # and str.splitlines() would break a row there (real Enron mail contains them).
    return [model.model_validate_json(line) for line in path.read_text().split("\n") if line]


def write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    path.write_text("".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows))
