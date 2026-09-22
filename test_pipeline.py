"""Offline tests: the Jev and Claude code paths run end to end against fakes, so a broken schema or
a renamed field fails here instead of 300 paid requests into an eval.

    uv run pytest -q
"""

import asyncio
import json
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.typesafe import TypeSafeProvider
from typesafe_sdk import AsyncTypeSafeClient
from typesafe_sdk._core.response_types import SystemOneResponse

import contestants
from labels import CATEGORIES, HAM, SCENARIOS, SPAM, Routing, ScenarioSv, SpamVerdict
from scorers import correct, recall_by_class, spam_precision


class Hooks:
    """The one method of Braintrust's EvalHooks the tasks use."""

    def __init__(self):
        self.metadata: dict = {}

    def meta(self, **kwargs):
        self.metadata |= kwargs


class FakeJev(AsyncTypeSafeClient):
    """Answers every question with canned answers and keeps what it was asked."""

    def __init__(self, answers: dict):
        super().__init__(api_key="test")
        self.answers = answers
        self.asked: list[tuple] = []

    async def system_one(self, state, questions, **kwargs):
        self.asked.append((state, questions, kwargs))
        return SystemOneResponse.model_validate(
            {"model": "jev-1.13.0", "usage": {"input_tokens": 42, "output_tokens": 1}, "answers": self.answers}
        )


def jev_agent[O: BaseModel](client: FakeJev, output_type: type[O]) -> Agent[Any, O]:
    return contestants._jev(output_type, TypeSafeModel("jev-latest", provider=TypeSafeProvider(typesafe_client=client)))


def test_jev_routing_sends_every_category_as_a_criterion():
    client = FakeJev(
        {
            "category": {
                "type": "choice",
                "choice": "refund",
                "confidence": 0.91,
                "probabilities": {"refund": 0.93, "order": 0.07},
            }
        }
    )
    task = contestants._agent_task(lambda: jev_agent(client, Routing), "category", contestants._route_label)
    hooks = Hooks()

    out = asyncio.run(task({"text": "where is my money back for order 123?"}, hooks))

    assert out == {"label": "refund", "confidence": 0.91}
    state, questions, _ = client.asked[0]
    assert "where is my money back" in json.dumps(state)
    assert questions["category"].criteria == CATEGORIES
    assert hooks.metadata["model_version"] == "jev-1.13.0"
    assert hooks.metadata["provider_details"]["probabilities"]["category"]["refund"] == 0.93


def test_jev_spam_is_a_yes_no_question():
    client = FakeJev({"is_spam": {"type": "noul", "noul": 0.97}})
    task = contestants._agent_task(lambda: jev_agent(client, SpamVerdict), "is_spam", contestants._spam_label)

    out = asyncio.run(task({"text": "CLICK HERE for cheap meds"}, Hooks()))

    assert out["label"] == SPAM
    assert out["confidence"] > 0.9
    assert client.asked[0][1]["is_spam"].type == "noul"


def test_jev_swedish_scenarios_send_english_criteria_with_swedish_state():
    client = FakeJev(
        {"scenario": {"type": "choice", "choice": "alarm", "confidence": 0.9, "probabilities": {"alarm": 0.9}}}
    )
    task = contestants._agent_task(lambda: jev_agent(client, ScenarioSv), "scenario", contestants._scenario_label)

    out = asyncio.run(task({"text": "väck mig klockan sju"}, Hooks()))

    assert out["label"] == "alarm"
    assert client.asked[0][1]["scenario"].criteria == SCENARIOS


def anthropic_agent(reply: dict, seen: list) -> Agent[Any, Routing]:
    """A real Anthropic client whose HTTP transport returns `reply` as the model's JSON text."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": json.dumps(reply)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 300, "output_tokens": 12},
            },
        )

    client = AsyncAnthropic(api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return contestants._llm(
        Routing, AnthropicModel("claude-opus-5", provider=AnthropicProvider(anthropic_client=client))
    )


def test_claude_routing_uses_structured_output_with_the_same_schema():
    seen: list = []
    task = contestants._agent_task(
        lambda: anthropic_agent({"category": "invoice"}, seen), "category", contestants._route_label
    )
    hooks = Hooks()

    out = asyncio.run(task({"text": "I need a copy of last month's bill"}, hooks))

    assert out == {"label": "invoice", "confidence": None}
    request = seen[0]
    schema_text = json.dumps(request["output_config"]["format"])
    assert all(description in schema_text for description in CATEGORIES.values())
    assert request["output_config"]["effort"] == "low"
    assert hooks.metadata["input_tokens"] == 300


@pytest.mark.skipif(
    not (contestants.ROOT / "models" / "route" / "model-best").exists(),
    reason="needs a trained model: uv run python data.py && uv run python train_spacy.py route",
)
def test_spacy_contestant_returns_a_known_label():
    spacy = next(c for c in contestants.contestants() if c.name == "spacy")
    hooks = Hooks()

    out = spacy.tasks["route"]({"text": "how do I download my invoice?"}, hooks)
    assert isinstance(out, dict)  # spaCy's tasks are synchronous

    assert out["label"] == "invoice"
    assert set(hooks.metadata["probabilities"]) == set(CATEGORIES)


def test_scorers_give_precision_and_recall_through_skipped_rows():
    said_spam, said_ham = {"label": SPAM}, {"label": HAM}
    assert correct(input={}, output=said_spam, expected=SPAM).score == 1.0
    assert recall_by_class(input={}, output=said_ham, expected=SPAM).name == "recall · spam"
    assert recall_by_class(input={}, output=said_ham, expected=SPAM).score == 0.0
    false_positive = spam_precision(input={}, output=said_spam, expected=HAM)
    assert false_positive is not None and false_positive.score == 0.0
    assert spam_precision(input={}, output=said_ham, expected=SPAM) is None  # predicted ham: not in precision


def test_records_reject_rows_that_do_not_fit_the_shape():
    import pydantic

    from records import Example

    assert Example.model_validate_json('{"message_id": "1", "text": "hej", "label": "email"}').label == "email"
    for bad in (
        '{"message_id": 11935, "text": "hej", "label": "email"}',  # an int id (Enron's, before the fix)
        '{"message_id": "1", "text": "hej"}',  # no label
        '{"message_id": "1", "text": "hej", "label": "email", "lable": "x"}',  # a typo'd key
    ):
        with pytest.raises(pydantic.ValidationError):
            Example.model_validate_json(bad)


def test_jsonl_round_trip_keeps_unicode_line_separators(tmp_path):
    from records import Example, read_jsonl, write_jsonl

    rows = [Example(message_id="1", text="first\u2028second\x1cthird\x85", label="ham")]
    write_jsonl(tmp_path / "rows.jsonl", rows)
    assert read_jsonl(tmp_path / "rows.jsonl", Example) == rows
