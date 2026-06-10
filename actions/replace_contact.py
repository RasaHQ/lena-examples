from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import Contact, get_contacts, write_contacts


class ReplaceContact(Action):
    def name(self) -> str:
        return "replace_contact"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        contacts = get_contacts(tracker.sender_id)
        old_handle = tracker.get_slot("replace_contact_old_handle")
        new_handle = tracker.get_slot("replace_contact_new_handle")
        new_name = tracker.get_slot("replace_contact_new_name")

        if old_handle is None or new_handle is None or new_name is None:
            return [SlotSet("return_value", "missing_data")]

        old_contact_index = next(
            (i for i, contact in enumerate(contacts) if contact.handle == old_handle), None
        )
        if old_contact_index is None:
            return [SlotSet("return_value", "not_found")]

        conflicting_contact = next(
            (
                contact
                for contact in contacts
                if contact.handle == new_handle and contact.handle != old_handle
            ),
            None,
        )
        if conflicting_contact is not None:
            return [SlotSet("return_value", "already_exists")]

        old_contact = contacts[old_contact_index]
        contacts[old_contact_index] = Contact(name=new_name, handle=new_handle)
        write_contacts(tracker.sender_id, contacts)

        return [
            SlotSet("return_value", "success"),
            SlotSet("replace_contact_old_name", old_contact.name),
        ]
