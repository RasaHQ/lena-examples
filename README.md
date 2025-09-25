# Action server error handling

[ZD ticket](https://rasahq.zendesk.com/agent/tickets/1914)

## Problem

Recommendations for error handling in custom actions in a conversational way. For example, if there is an error with the API request, then bot should say something went wrong and try again later.

## Notes

- CALM doesn't do anything OOTB when a custom action encounters an exception. 
- CALM continues on with the flow so you can log and raise an exception but CALM will carry on.
- You can set a slot like `custom_action_error` and use that to branch to link a flow or `pattern_internal_error` but then you'd have to do that at every step there is a custom action. This doesn't seem scalable.

What Albert Heijn does:

- action server has an error
- send message to user about there being an error
- hand over the customer and end the conversation
- followup action `action_clean_stack` in custom action
- conversation processor changes handler to human

## Ideas

### 1- do it in flows
set slot `custom_action_error` to `True` and branch in flow

**Pros:**

- Logic is in flow
- can link to flow (human handoff)

**Cons:**

- have to add to every step that has a custom action

### 2 - do it in custom action (no pattern triggering)
followup `action_clean_stack` in the custom action to clean the stack. 
add `action_listen` so `pattern_completed` does not trigger (??? untested)

**Pros**:
- All in custom action. 
- Could make a custom `Action` class to make this easier to scale.

**Cons**:

- All logic is in custom action. 


## 3 - start pattern_internal_error in custom action

Add nlu_trigger to pattern_internal_error and trigger intent from custom action (??? untested)

**Pros**
- easier to add branching logic to patterns
- logic is in flow

**Cons**

- unexpected behaviour with `patttern_continue_interrupted`?

## Bot details

- `pattern_session_start` has been modified to allow you to set a slot to trigger error in `list_contact` flow
- there is a slot `custom_action_error` that will be true if the custom action fails. Branch on that slot. How to scale?

This is the ideal conversation. Maybe better without `pattern_completed` (not sure how to do that yet)
```
<--- convo starts --->
Bot: Do you want to test error handling?
User: Yes
<bot listening>
User: list contacts
<list_contacts custom action runs and errors>
Bot: Sorry, I am having trouble with that. Please try again in a few minutes.
Bot: <pattern_completed>
<--- convo ends --->
```
