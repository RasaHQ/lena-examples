"""Custom NLU command adapter with low-confidence flow fallback.

Usage in `config.yml`:

pipeline:
  - name: examples.low_confidence_fallback_nlu_command_adapter.
      LowConfidenceFallbackNLUCommandAdapter
    fallback_flow_id: two_stage_fallback
    enabled: true
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from rasa.dialogue_understanding.commands import (
    Command,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.error_command import ErrorCommand
from rasa.dialogue_understanding.generator.nlu_command_adapter import (
    NLUCommandAdapter,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.shared.nlu.training_data.message import Message

structlogger = structlog.get_logger(__name__)


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.COMMAND_GENERATOR],
    is_trainable=False,
)
class FallbackNLUCommandAdapter(NLUCommandAdapter):
    """Start one fallback flow when an NLU trigger is a near-miss.

    This class only overrides behavior-specific methods (`get_default_config`,
    `__init__`, and `predict_commands`). Component lifecycle methods (`create`,
    `load`, `train`) are inherited from `NLUCommandAdapter`.
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "fallback_flow_id": None,
            "enabled": True,
        }

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> None:
        super().__init__(config, model_storage, resource, execution_context)
        self._fallback_flow_id: Optional[str] = self.config.get(
            "fallback_flow_id"
        )
        self._enabled: bool = bool(self.config.get("enabled", True))

    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        **kwargs: Any,
    ) -> List[Command]:
        commands = await super().predict_commands(
            message, flows, tracker, **kwargs
        )

        matched_flow_id = self._find_low_confidence_trigger_match(message, flows)
        if not self._should_inject_fallback(
            commands, matched_flow_id, flows, tracker
        ):
            return commands

        # Replace all previously predicted commands with the fallback flow start.
        fallback_commands: List[Command] = [StartFlowCommand(self._fallback_flow_id)]
        if tracker is not None and tracker.has_coexistence_routing_slot:
            fallback_commands.append(SetSlotCommand(ROUTE_TO_CALM_SLOT, True))

        return fallback_commands

    def _should_inject_fallback(
        self,
        commands: List[Command],
        matched_flow_id: Optional[str],
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker],
    ) -> bool:
        """Decide whether to replace predicted commands with the fallback flow.

        The fallback is injected only when **all** of the following hold:
          1. the component is enabled and a fallback flow id is configured,
          2. the tracker exists and flows are non-empty,
          3. the message is a near-miss on some flow's NLU trigger,
          4. the configured fallback flow id exists in the loaded flows,
          5. no other StartFlowCommand or ErrorCommand was already predicted
             (we do not stack fallbacks on top of an existing decision),
          6. the conversation is not currently in a collect step
             (we must not interrupt slot collection).
        """
        if not self._is_configured():
            return False
        if tracker is None or flows.is_empty():
            return False
        if matched_flow_id is None:
            return False
        if not self._fallback_flow_exists(flows):
            return False
        if self._has_blocking_command(commands):
            return False
        if self._is_in_collect_step(tracker, flows):
            structlogger.debug(
                "low_confidence_fallback_nlu_adapter.skip_in_collect_step",
                fallback_flow_id=self._fallback_flow_id,
            )
            return False
        return True

    def _is_configured(self) -> bool:
        return self._enabled and bool(self._fallback_flow_id)

    def _fallback_flow_exists(self, flows: FlowsList) -> bool:
        if self._fallback_flow_id in flows.flow_ids:
            return True
        structlogger.warning(
            "low_confidence_fallback_nlu_adapter.unknown_fallback_flow",
            fallback_flow_id=self._fallback_flow_id,
        )
        return False

    @staticmethod
    def _has_blocking_command(commands: List[Command]) -> bool:
        return any(
            isinstance(command, (StartFlowCommand, ErrorCommand))
            for command in commands
        )

    @staticmethod
    def _is_in_collect_step(
        tracker: DialogueStateTracker, flows: FlowsList
    ) -> bool:
        from rasa.dialogue_understanding.processor.command_processor import (
            get_current_collect_step,
        )

        return get_current_collect_step(tracker.stack, flows) is not None

    def _find_low_confidence_trigger_match(
        self, message: Message, flows: FlowsList
    ) -> Optional[str]:
        """Find a flow whose trigger intent matches but confidence is too low.

        This method returns the first matching *flow id* so callers can use it
        for richer logging/metrics later. It does not mutate commands; it only
        answers whether the message is in the "near miss" band:
        - same intent name as a flow's trigger
        - confidence below that trigger's confidence threshold
        """
        intent_data = message.get(INTENT)
        if not intent_data or not intent_data.get(INTENT_NAME_KEY):
            return None

        intent_name = intent_data[INTENT_NAME_KEY]
        confidence = float(intent_data.get(PREDICTED_CONFIDENCE_KEY, 0.0))

        for flow in flows:
            # Skip system default flows and fallback itself to avoid loops.
            if flow.is_rasa_default_flow or flow.id == self._fallback_flow_id:
                continue
            if not flow.nlu_triggers:
                continue

            for trigger in flow.nlu_triggers.trigger_conditions:
                if (
                    trigger.intent == intent_name
                    and confidence < trigger.confidence_threshold
                ):
                    structlogger.debug(
                        "low_confidence_fallback_nlu_adapter."
                        "low_confidence_trigger_match",
                        matched_flow_id=flow.id,
                        intent=intent_name,
                        confidence=confidence,
                        threshold=trigger.confidence_threshold,
                    )
                    return flow.id

        return None
