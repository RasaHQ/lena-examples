from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_contacts


class ListContacts(Action):
    def name(self) -> str:
        return "list_contacts"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        contacts = get_contacts(tracker.sender_id)
        if len(contacts) > 0:
            # Spoken list for TTS: "Alice at @alice, Bob at @bob."
            spoken = [
                f"{contact.name} at {contact.handle}" for contact in contacts
            ]
            if len(spoken) == 1:
                contacts_list = f"{spoken[0]}."
            else:
                contacts_list = (
                    f"{', '.join(spoken[:-1])}, and {spoken[-1]}."
                )
            return [SlotSet("contacts_list", contacts_list)]
        else:
            return [SlotSet("contacts_list", None)]
