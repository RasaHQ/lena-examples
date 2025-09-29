from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, FollowupAction, UserUttered
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_contacts
import logging
import traceback
logger = logging.getLogger(__name__)

from actions.safeaction import SafeAction

class ListContacts(SafeAction):
    def name(self) -> str:
        return "list_contacts"

    def safe_run(self, dispatcher, tracker, domain) -> List[Dict[Text, Any]]:
        # # Add test flag to trigger exception
        # test_error = tracker.get_slot("test_error")
        # if test_error:
        #     logger.error("Test exception in list_contacts")
        #     dispatcher.utter_message(response="utter_internal_error_rasa")
        #     return [SlotSet("action_server_error", True), 
        #             FollowupAction("action_clean_stack")]

        # try:
        #     contacts = get_contacts(tracker.sender_id)
        #     if len(contacts) > 0:
        #         contacts_list = "".join([f"- {c.name} ({c.handle}) \n" for c in contacts])
        #         return [SlotSet("contacts_list", contacts_list)]
        #     else:
        #         return [SlotSet("contacts_list", None)]
        # except Exception as e:
        #     logger.error(f"Exception in list_contacts: {e}")
        #     dispatcher.utter_message(response="utter_internal_error_rasa")
        #     return [SlotSet("contacts_list", None), 
        #             SlotSet("action_server_error", True),
        #             FollowupAction("action_clean_stack")]
        contacts = get_contacts(tracker.sender_id)
        if len(contacts) > 0:
            contacts_list = "".join([f"- {c.name} ({c.handle}) \n" for c in contacts])
            return [SlotSet("contacts_list", contacts_list)]
        else:
            return [SlotSet("contacts_list", None)]
        