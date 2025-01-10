"""Intent override module.

This module provides custom component for overriding intents.
"""

import logging
from typing import Any

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message

# ruff: noqa: ARG002, ARG003

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER],
    is_trainable=False,
)
class IntentOverrideComponent(GraphComponent):  # type: ignore[misc]
    """Custom NLU component to override intents based on metadata."""

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        """Constructor for component."""
        self.config = config
        logger.info("---- BUILDING overriding intent component ------")

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "IntentOverrideComponent":
        """Creates the component with the given configuration."""
        logger.info("---- CREATING overriding intent component ------")
        return cls(config)

    def process(self, messages: list[Message]) -> list[Message]:
        """Processes message using custom intents."""
        for message in messages:
            logger.info(f"----- OVERRIDING component message: {message!s} ")

            override_intent = "greet"
            message.set(
                "intent",
                {"name": override_intent, "confidence": 1.0},
                add_to_output=True,
            )
            message.set(
                "intent_ranking",
                [{"name": override_intent, "confidence": 1.0}],
                add_to_output=True,
            )
        return messages
