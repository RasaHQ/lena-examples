from argparse import Action
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging

CLIENT_AUTH_HEADER_KEY = "client_auth_headers"

# Configure logger
logger = logging.getLogger(__name__)

# Default headers to extract and propagate
HEADERS_TO_PROPAGATE = [
    "Accept-Language",
    "Authorization",
    "User-Agent",
    "X-Auth-Token"
]


def get_client_auth_headers(tracker: Tracker) -> Dict[str, str]:
    """
    Extract authentication headers from the tracker's metadata.

    Args:
        tracker: Rasa tracker containing conversation metadata

    Returns:
        Dictionary containing authentication headers
    """
    # headers = {
    #     "User-Agent": "Rasa Action Server",
    #     "Accept": "application/json",
    #     "Content-Type": "application/json",
    # }

    try:
        metadata = tracker.latest_message.get("metadata", {})
        headers = metadata.get("headers", {})

        return headers
    except Exception as e:
        logger.exception(f"Error extracting client auth headers: {e}")
        return headers


class ActionReadHeaders(Action):
    def name(self) -> Text:
        return "action_read_headers"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """
        Example action that uses extracted headers to call an external API.

        Args:
            dispatcher: Rasa dispatcher to send messages back to the user
            tracker: Rasa tracker containing conversation state
            domain: Rasa domain configuration

        Returns:
            List of events (empty in this case)
        """
        headers = get_client_auth_headers(tracker)
        logging.info(f"Extracted headers: {headers}")

        text = "\n".join([f"{key}: {value}" for key, value in headers.items()])

        return dispatcher.utter_message(text=f"Headers used for API call:\n{text}")
