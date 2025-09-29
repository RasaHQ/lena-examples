from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.events import FollowupAction
from rasa_sdk.executor import CollectingDispatcher
from abc import ABC, abstractmethod

import logging
import traceback
logger = logging.getLogger(__name__)

# Centralized error handling
class SafeAction(Action,ABC):

    @abstractmethod
    def name(self) -> str:
        print(f'''[ERROR] safeAction.name() called instead of subclass!''')
        traceback.print_stack()
        raise NotImplementedError('''You must define the "name" method''')
    
    @abstractmethod
    def safe_run(self
                 ,dispatcher: CollectingDispatcher
                 ,tracker: Tracker
                 , domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        raise NotImplementedError('''You must implement "safe_run" in your action''')
    
    def run(self
            ,dispatcher: CollectingDispatcher
            ,tracker: Tracker
            , domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            return self.safe_run(dispatcher, tracker, domain)
        except Exception as e:
            # logging the error
            logger.error(f'''[Error in {self.name}]: {str(e)}\n{traceback.format_exc()}''')
            # default fallback
            dispatcher.utter_message(response="utter_internal_error_rasa")
            return [FollowupAction("action_clean_stack")]

