"""The three classifiers under test, each exposing the same two tasks with the same output shape.

    spacy   the textcat pipelines from train_spacy.py, local, supervised on the task's own data
    spacy-<variant>  the same pipeline trained on less or different data: -20shot, or distill.py's
                     -gold2000 / -jev2000 / -jev80_2000 (see distill.py)
    jev     TypeSafe's Jev via Pydantic AI, zero-shot, reading the schemas in labels.py
    claude  a general LLM via Pydantic AI, zero-shot, with the very same schemas as structured output

Tasks: spam, route (English) and scenario_sv (Swedish). A spaCy contestant runs the tasks it has a
trained model for; Jev and Claude run all three.

Every task returns `{"label": str, "confidence": float | None}` and records model-specific detail
(probabilities, model version, tokens) in the Braintrust span metadata via `hooks.meta`.
"""

import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModelSettings

from labels import HAM, SPAM, Routing, ScenarioSv, SpamVerdict
from settings import settings

ROOT = Path(__file__).parent
JEV_MODEL = settings.jev_model
LLM_MODEL = settings.llm_model

Prediction = dict[str, Any]
Task = Callable[[dict, Any], Prediction | Awaitable[Prediction]]


@dataclass(frozen=True)
class Contestant:
    name: str
    model: str
    tasks: dict[str, Task]
    needs: str | None = None  # the env var holding its API key, if it calls one

    @property
    def missing_key(self) -> bool:
        return self.needs is not None and not os.getenv(self.needs)


# --- spaCy ---------------------------------------------------------------------------------------


@cache  # one pipeline per task per process: loading is ~100 ms, a prediction is well under 1 ms
def _pipeline(name: str):
    import spacy

    path = ROOT / "models" / name / "model-best"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `uv run python train_spacy.py` first")
    return spacy.load(path)


def _spacy(name: str) -> Task:
    def run(input: dict, hooks) -> Prediction:
        cats = _pipeline(name)(input["text"]).cats
        label = max(cats, key=cats.__getitem__)
        hooks.meta(probabilities={k: round(v, 4) for k, v in cats.items()})
        return {"label": label, "confidence": cats[label]}

    return run


# --- Pydantic AI (Jev, Claude) -------------------------------------------------------------------


def _agent_task(make_agent: Callable[[], Agent], field: str, to_label: Callable[[BaseModel], str]) -> Task:
    # Built on first use, not at import: a provider without its API key refuses to construct, and
    # a contestant whose key is missing is skipped rather than failing the whole run.
    agent = cache(make_agent)

    async def run(input: dict, hooks) -> Prediction:
        result = await agent().run(input["text"])
        details = result.response.provider_details or {}
        usage = result.usage
        hooks.meta(
            model_version=result.response.model_name,
            provider_details=details,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return {"label": to_label(result.output), "confidence": details.get("confidence", {}).get(field)}

    return run


def _spam_label(verdict: SpamVerdict) -> str:
    return SPAM if verdict.is_spam else HAM


def _route_label(routing: Routing) -> str:
    return routing.category


def _scenario_label(request: ScenarioSv) -> str:
    return request.scenario


# task -> (output schema, the field Jev's confidence is reported under, output -> label)
SCHEMAS = {
    "spam": (SpamVerdict, "is_spam", _spam_label),
    "route": (Routing, "category", _route_label),
    "scenario_sv": (ScenarioSv, "scenario", _scenario_label),
}


def _jev(output_type: type[BaseModel], model: Model | str = JEV_MODEL) -> Agent:
    return Agent(model, output_type=output_type)


def _llm(output_type: type[BaseModel], model: Model | str = LLM_MODEL) -> Agent:
    # NativeOutput is the API's structured-output mode, so the reply is always schema-valid JSON,
    # the same guarantee Jev makes. Low effort: a one-field classification needs little thinking,
    # and it keeps the latency comparison with Jev honest about what an LLM costs at its cheapest.
    return Agent(
        model,
        output_type=NativeOutput(output_type),
        model_settings=AnthropicModelSettings(anthropic_effort="low", max_tokens=2048),
    )


def _spacy_tasks(suffix: str = "") -> dict[str, Task]:
    """A spaCy task for every schema task with a trained `models/<task><suffix>`."""
    return {task: _spacy(task + suffix) for task in SCHEMAS if (ROOT / "models" / (task + suffix) / "model-best").exists()}


def _variant_suffixes() -> list[str]:
    """Every `-<variant>` with a trained `models/<task>-<variant>`: `-20shot` from train_spacy.py
    --shots 20, `-gold2000`/`-jev2000`/`-jev80_2000` from distill.py, and so on."""
    tasks = "|".join(sorted(SCHEMAS, key=len, reverse=True))
    found = (re.fullmatch(rf"({tasks})(-.+)", p.name) for p in (ROOT / "models").glob("*-*"))
    return sorted({m[2] for m in found if m})


_VARIANT_MODELS = {
    r"-(\d+)shot": "textcat ensemble, {} labelled per class",
    r"-gold(\d+)": "textcat ensemble, random pool of {} with gold labels",
    r"-jev(\d+)": "textcat ensemble, random pool of {} labelled by Jev",
    r"-jev80_(\d+)": "textcat ensemble, random pool of {}, Jev labels with confidence >= 0.8",
}


def _describe(suffix: str) -> str:
    return next(
        (text.format(m[1]) for pattern, text in _VARIANT_MODELS.items() if (m := re.fullmatch(pattern, suffix))),
        f"textcat ensemble{suffix}",
    )


def _agent_tasks(make_agent: Callable[[type[BaseModel]], Agent]) -> dict[str, Task]:
    return {
        task: _agent_task(lambda schema=schema: make_agent(schema), field, to_label)
        for task, (schema, field, to_label) in SCHEMAS.items()
    }


def contestants() -> list[Contestant]:
    return [
        Contestant("spacy", "textcat ensemble, full training split", _spacy_tasks()),
        *(Contestant(f"spacy{suffix}", _describe(suffix), _spacy_tasks(suffix)) for suffix in _variant_suffixes()),
        Contestant("jev", JEV_MODEL, _agent_tasks(_jev), needs="TYPESAFE_API_KEY"),
        Contestant(
            "claude",
            LLM_MODEL,
            _agent_tasks(_llm),
            needs="ANTHROPIC_API_KEY" if LLM_MODEL.startswith("anthropic:") else None,
        ),
    ]
