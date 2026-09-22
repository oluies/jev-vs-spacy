"""The two tasks' label sets and output schemas, the single definition every contestant shares.

Jev reads these schemas as its prompt: the model docstring is the goal, a field description is
the question, and each option's description is its criterion. Claude gets the same schema as a
structured output. spaCy never sees the prose; it learns the same labels from training data.
"""

from typing import Annotated, Literal, Union, get_args

from pydantic import BaseModel, Field

SPAM, HAM = "spam", "ham"

# Bitext's eleven customer-support categories, lower-cased, each described the way a support lead
# would brief a new agent. The descriptions follow the dataset's intents so the labels are learnable.
CATEGORIES: dict[str, str] = {
    "account": ("Creating, editing, deleting, switching or recovering a user account, or a registration problem."),
    "order": "Placing, changing or cancelling an order, or tracking where an order is now.",
    "refund": "Asking for a refund, the refund policy, or the status of a refund.",
    "invoice": "Finding, viewing or downloading an invoice.",
    "payment": "Which payment methods are accepted, or a payment that failed.",
    "delivery": (
        "When a purchase will arrive (delivery time or date), "
        "or which delivery options exist and where the company delivers."
    ),
    "shipping": "Setting up or changing the shipping address.",
    "cancel": "Fees, charges or penalties for cancelling or withdrawing, not the act of cancelling an order.",
    "subscription": "Subscribing to or unsubscribing from the newsletter.",
    "feedback": "Leaving a review, or filing a complaint or consumer claim about the service.",
    "contact": "Reaching customer service or a human agent.",
}

# One `Literal` per category, each carrying its description, so the JSON schema is an `anyOf` of
# `{const, description}` and Jev gets a criterion per option (a bare `Literal`/`Enum` would not).
# Built at runtime from the dict so the labels and their descriptions have one source; a static
# checker cannot follow that, and `X | Y` cannot take a tuple, hence the two suppressions.
Category = Union[tuple(Annotated[Literal[name], Field(description=text)] for name, text in CATEGORIES.items())]  # noqa: UP007  # ty: ignore[invalid-type-form]

assert [get_args(get_args(member)[0])[0] for member in get_args(Category)] == list(CATEGORIES)


class SpamVerdict(BaseModel):
    """Screen an email that arrived in an employee's corporate inbox."""

    is_spam: bool = Field(
        description=(
            "Is this unsolicited bulk email (advertising, scams, phishing or adult content) "
            "rather than mail from someone the recipient actually deals with?"
        )
    )


class Routing(BaseModel):
    """Route a customer's support message to the team that handles it."""

    category: Category = Field(description="What is the customer asking about?")  # ty: ignore[invalid-type-form]


# MASSIVE's eighteen voice-assistant scenarios (Swedish split). The requests are in Swedish; the
# criteria stay in English, which is how a Swedish team would usually write a schema.
SCENARIOS: dict[str, str] = {
    "alarm": "Setting, checking or removing an alarm.",
    "audio": "The device's volume: louder, quieter or mute.",
    "calendar": "Calendar events and reminders: adding, checking or removing them.",
    "cooking": "Recipes and how to cook something.",
    "datetime": "The current date or time, or the time somewhere else.",
    "email": "Reading, writing or managing email, or email contacts.",
    "general": "Small talk, jokes, greetings, or asking the assistant to repeat or confirm.",
    "iot": "Controlling smart-home devices: lights, plugs, the robot vacuum, the coffee machine.",
    "lists": "Creating, reading or changing a list, such as a shopping or to-do list.",
    "music": "Music settings and info: what is playing, liking a song, shuffle, music preferences.",
    "news": "News headlines or news about a topic.",
    "play": "Starting playback of music, radio, a podcast, an audiobook or a game.",
    "qa": "Factual questions: definitions, facts, arithmetic, stock prices, exchange rates.",
    "recommendation": "Recommendations for events, places to go or films to watch.",
    "social": "Posting to or reading social media, including complaints to companies there.",
    "takeaway": "Ordering takeaway food or checking a takeaway order.",
    "transport": "Tickets, taxis, trains and traffic.",
    "weather": "The weather or the forecast.",
}

Scenario = Union[tuple(Annotated[Literal[name], Field(description=text)] for name, text in SCENARIOS.items())]  # noqa: UP007  # ty: ignore[invalid-type-form]


class ScenarioSv(BaseModel):
    """Route a Swedish request to a voice assistant to the skill that handles it."""

    scenario: Scenario = Field(description="Which skill should handle this request?")  # ty: ignore[invalid-type-form]
