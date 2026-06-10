# custom_tracker_store.py

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional, Set

from rasa.core.tracker_stores.sql_tracker_store import SQLTrackerStore
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


class ScrubbingSQLTrackerStore(SQLTrackerStore):
    """Redacts configured auth fields before SQL persistence and event streaming."""

    DEFAULT_SENSITIVE_AUTH_FIELDS: Set[str] = {
        "access_token",
        "auth_token",
        "id_token",
        "jwt",
        "refresh_token",
    }

    def __init__(
        self,
        *args: Any,
        sensitive_auth_fields: Optional[Iterable[str]] = None,
        replacement: str = "[REDACTED]",
        **kwargs: Any,
    ) -> None:
        self.sensitive_auth_fields = {
            field.lower()
            for field in (
                sensitive_auth_fields or self.DEFAULT_SENSITIVE_AUTH_FIELDS
            )
        }
        self.replacement = replacement

        super().__init__(*args, **kwargs)

    async def save(self, tracker: DialogueStateTracker) -> None:
        scrubbed_tracker = copy.deepcopy(tracker)
        self._scrub_tracker(scrubbed_tracker)

        await super().save(scrubbed_tracker)

    def _scrub_tracker(self, tracker: DialogueStateTracker) -> None:
        for slot_name in list(tracker.slots.keys()):
            if slot_name.lower() in self.sensitive_auth_fields:
                tracker.slots[slot_name] = self.replacement

        for event in tracker.events:
            self._scrub_event(event)

    def _scrub_event(self, event: Event) -> None:
        if isinstance(event, SlotSet) and event.key.lower() in self.sensitive_auth_fields:
            event.value = self.replacement

        if hasattr(event, "metadata"):
            event.metadata = self._scrub_metadata(event.metadata)

    def _scrub_metadata(self, metadata: Any) -> Any:
        if not isinstance(metadata, dict):
            return metadata

        scrubbed = copy.deepcopy(metadata)

        # Direct metadata auth values, e.g. {"access_token": "..."}.
        for key in list(scrubbed.keys()):
            if str(key).lower() in self.sensitive_auth_fields:
                scrubbed[key] = self.replacement

        # Metadata that mirrors slot values, e.g. {"slots": {"access_token": "..."}}.
        slots = scrubbed.get("slots")
        if isinstance(slots, dict):
            for slot_name in list(slots.keys()):
                if str(slot_name).lower() in self.sensitive_auth_fields:
                    slots[slot_name] = self.replacement

        return scrubbed