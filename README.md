# Action server error handling

[ZD ticket](https://rasahq.zendesk.com/agent/tickets/1914)

### Problem

Recommendations for error handling in custom actions in a conversational way. For example, if there is an error with the API request, then bot should say something went wrong and try again later.

### Findings

- CALM doesn't do anything OOTB when a custom action encounters an exception. 
- CALM continues on with the flow so you can log and raise an exception but CALM will carry on.
- You can set a slot like `custom_action_error` and use that to branch to link a flow or `pattern_internal_error` but then you'd have to do that at every step there is a custom action. This doesn't seem scalable.

### Ideas

- use followup `action_clean_stack` in the custom action clean the stack

What Albert Heijn does:

- action server has an error
- send message to user about there being an error
- hand over the customer and end the conversation
- followup action `action_clean_stack` in custom action
- conversation processor changes handler to human

### Bot details

- `pattern_session_start` has been modified to allow you to set a slot to trigger error in `list_contact` flow
- there is a slot `custom_action_error` that will be true if the custom action fails. Branch on that slot. How to scale?