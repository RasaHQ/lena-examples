## bot_challenge

This flow clarifies that the user is talking to a chatbot, when the user ask if they are talking to a chatbot or a human

```mermaid
graph TD
    utter_bot_challenge[utter_bot_challenge]
```

## leave_feedback

This flow collects simple thumbs up or thumbs down feedback

**Flow Guard:** if: False (only triggered via call/link)

```mermaid
graph TD
    END([END])
    feedback_rating[collect: feedback_rating]
    feedback_response[feedback_response]
    positive_response[utter_thankyou_positive]
    negative_response[utter_thankyou_negative]
    feedback_rating --> feedback_response
    feedback_response -->|if slots.feedback_rating == thumbs_up| positive_response
    feedback_response -->|else| negative_response
    positive_response --> END
    negative_response --> END
```

## goodbye

Handle user goodbye messages, say farewell, and collect feedback

```mermaid
graph TD
    utter_goodbye[utter_goodbye]
    leave_feedback{{link: leave_feedback}}
    utter_goodbye --> leave_feedback
```

## hello

Flow for greeting the user hello

```mermaid
graph TD
    utter_hello[utter_hello]
```

## help

This flow addresses user inquiries about the agent's capabilities. It starts by acknowledging the user's request and providing a clear overview of key skills and services. This helps users understand how the agent can assist them effectively.

```mermaid
graph TD
    utter_help[utter_help]
```

## human_handoff

When you are unable to handle the request or the user is frustrated - offers to hand the user over to a live human agent.

```mermaid
graph TD
    END([END])
    confirm_human_handoff[collect: confirm_human_handoff]
    utter_transferring_to_human[utter_transferring_to_human]
    action_human_handoff[action_human_handoff]
    utter_human_handoff_cancelled[utter_human_handoff_cancelled]
    utter_transferring_to_human --> action_human_handoff
    confirm_human_handoff -->|if slots.confirm_human_handoff| utter_transferring_to_human
    confirm_human_handoff -->|else| utter_human_handoff_cancelled
    action_human_handoff --> END
    utter_human_handoff_cancelled --> END
```

## welcome

This flow is designed to greet and onboard new users who initiate a welcome intent.
It begins with an initial greeting message to establish a friendly interaction.


**Flow Guard:** if: False (only triggered via call/link)

```mermaid
graph TD
    utter_greeting[utter_greeting]
```

## activate_card

Activates user's card.

```mermaid
graph TD
    END([END])
    utter_start_activate_card[utter_start_activate_card]
    get_bank_card_number[collect: get_bank_card_number]
    check_that_card_exists[check_that_card_exists]
    utter_bank_card_not_found[utter_bank_card_not_found]
    activation_confirmation[collect: activation_confirmation]
    utter_activate_card_cancelled[utter_activate_card_cancelled]
    execute_activation[utter_activate_card_successful]
    utter_start_activate_card --> get_bank_card_number
    get_bank_card_number --> check_that_card_exists
    check_that_card_exists -->|if not slots.return_value| utter_bank_card_not_found
    check_that_card_exists -->|else| activation_confirmation
    utter_bank_card_not_found --> get_bank_card_number
    activation_confirmation -->|if not slots.confirmation| utter_activate_card_cancelled
    activation_confirmation -->|else| execute_activation
    utter_activate_card_cancelled --> END
    execute_activation --> END
```

## block_card

Applies a permanent or temporary block to stolen, lost or compromised banking cards.

```mermaid
graph TD
    END([END])
    utter_start_block_card[utter_start_block_card]
    get_bank_card_number[collect: get_bank_card_number]
    check_that_card_exists[check_that_card_exists]
    utter_bank_card_not_found[utter_bank_card_not_found]
    permanent_or_temporary_block[collect: permanent_or_temporary_block]
    confirm_block_card[collect: confirm_block_card]
    utter_block_card_cancelled[utter_block_card_cancelled]
    execute_permanent_block[utter_block_card_successful]
    execute_temporary_freeze[utter_freeze_card_successful]
    utter_start_block_card --> get_bank_card_number
    get_bank_card_number --> check_that_card_exists
    permanent_or_temporary_block --> confirm_block_card
    check_that_card_exists -->|if not slots.return_value| utter_bank_card_not_found
    check_that_card_exists -->|else| permanent_or_temporary_block
    utter_bank_card_not_found --> get_bank_card_number
    confirm_block_card -->|if not slots.confirmation| utter_block_card_cancelled
    confirm_block_card -->|if slots.card_block_type is permanent| execute_permanent_block
    confirm_block_card -->|else| execute_temporary_freeze
    utter_block_card_cancelled --> END
    execute_permanent_block --> END
    execute_temporary_freeze --> END
```

## list_cards

show your card list

```mermaid
graph TD
    END([END])
    list_cards[list_cards]
    utter_list_cards[utter_list_cards]
    utter_no_cards[utter_no_cards]
    list_cards -->|if slots.cards_list| utter_list_cards
    list_cards -->|else| utter_no_cards
    utter_list_cards --> END
    utter_no_cards --> END
```

## replace_card

Replaces an existing card if it’s lost, stolen, damaged, or compromised (including suspected fraud).

```mermaid
graph TD
    END([END])
    confirm_correct_card[collect: confirm_correct_card]
    replace_eligible_card[[call: replace_eligible_card]]
    utter_relevant_card_not_linked[utter_relevant_card_not_linked]
    confirm_correct_card -->|if slots.confirm_correct_card| replace_eligible_card
    confirm_correct_card -->|else| utter_relevant_card_not_linked
    replace_eligible_card --> END
    utter_relevant_card_not_linked --> END
```

## replace_eligible_card

Guides the user through replacing an eligible card based on the reason (lost, damaged, or unknown)
and addresses potential fraud if needed


```mermaid
graph TD
    END([END])
    replacement_reason[collect: replacement_reason]
    was_card_used_fraudulently[collect: was_card_used_fraudulently]
    utter_report_fraud[utter_report_fraud]
    utter_unknown_replacement_reason_handover[utter_unknown_replacement_reason_handover]
    start_replacement[utter_will_cancel_and_send_new]
    utter_new_card_has_been_ordered[utter_new_card_has_been_ordered]
    start_replacement --> utter_new_card_has_been_ordered
    replacement_reason -->|if slots.replacement_reason == lost| was_card_used_fraudulently
    replacement_reason -->|if slots.replacement_reason == damaged| start_replacement
    replacement_reason -->|else| utter_unknown_replacement_reason_handover
    was_card_used_fraudulently -->|if slots.was_card_used_fraudulently| utter_report_fraud
    was_card_used_fraudulently -->|else| start_replacement
    utter_report_fraud --> END
    utter_unknown_replacement_reason_handover --> END
```

## add_contact

Add a person with their contact details to your contacts list to enable simplified payments.

```mermaid
graph TD
    END([END])
    add_contact_handle[collect: add_contact_handle]
    add_contact_name[collect: add_contact_name]
    confirmation[collect: confirmation]
    utter_add_contact_cancelled[utter_add_contact_cancelled]
    add_contact[add_contact]
    utter_contact_added[utter_contact_added]
    utter_contact_already_exists[utter_contact_already_exists]
    utter_add_contact_error[utter_add_contact_error]
    add_contact_handle --> add_contact_name
    add_contact_name --> confirmation
    confirmation -->|if not slots.confirmation| utter_add_contact_cancelled
    confirmation -->|else| add_contact
    utter_add_contact_cancelled --> END
    add_contact -->|if slots.return_value = 'success'| utter_contact_added
    add_contact -->|if slots.return_value = 'already_exists'| utter_contact_already_exists
    add_contact -->|else| utter_add_contact_error
    utter_contact_added --> END
    utter_contact_already_exists --> END
    utter_add_contact_error --> END
```

## list_contacts

show your contact list

```mermaid
graph TD
    END([END])
    list_contacts[list_contacts]
    utter_list_contacts[utter_list_contacts]
    utter_no_contacts[utter_no_contacts]
    list_contacts -->|if slots.contacts_list| utter_list_contacts
    list_contacts -->|else| utter_no_contacts
    utter_list_contacts --> END
    utter_no_contacts --> END
```

## remove_contact

Remove somebody and their details from your contacts list.

```mermaid
graph TD
    END([END])
    remove_contact_handle[collect: remove_contact_handle]
    confirmation[collect: confirmation]
    utter_remove_contact_cancelled[utter_remove_contact_cancelled]
    remove_contact[remove_contact]
    utter_remove_contact_success[utter_remove_contact_success]
    utter_contact_not_in_list[utter_contact_not_in_list]
    utter_remove_contact_error[utter_remove_contact_error]
    remove_contact_handle --> confirmation
    confirmation -->|if not slots.confirmation| utter_remove_contact_cancelled
    confirmation -->|else| remove_contact
    utter_remove_contact_cancelled --> END
    remove_contact -->|if slots.return_value = 'success'| utter_remove_contact_success
    remove_contact -->|if slots.return_value = 'not_found'| utter_contact_not_in_list
    remove_contact -->|else| utter_remove_contact_error
    utter_remove_contact_success --> END
    utter_contact_not_in_list --> END
    utter_remove_contact_error --> END
```

## check_transfer_limit

Provides the maximum allowed amount the user can send or transfer per transaction, per day, or per month, as set by account policies—not the current available balance.

```mermaid
graph TD
    END([END])
    transfer_limit_type[collect: transfer_limit_type]
    check_transfer_limit[check_transfer_limit]
    utter_transfer_limit_error[utter_transfer_limit_error]
    show_transfer_limit[utter_show_transfer_limit]
    transfer_limit_type --> check_transfer_limit
    check_transfer_limit -->|if not slots.return_value| utter_transfer_limit_error
    check_transfer_limit -->|else| show_transfer_limit
    utter_transfer_limit_error --> END
```

## list_payees

List your payees or authorised recipients for money transfers. 
Show the people in your contact list who are valid payees you can transfer money to.


```mermaid
graph TD
    END([END])
    list_payees[list_payees]
    list_payees --> END
```

## list_transactions

Enables users to quickly view their most recent account transactions. 
After presenting these, the users has the option to view their scheduled (upcoming) 
transactions.


```mermaid
graph TD
    END([END])
    list_transactions[list_transactions]
    utter_transactions[utter_transactions]
    list_transactions_next_option[collect: list_transactions_next_option]
    utter_no_next_option[utter_no_next_option]
    show_scheduled_transactions{{link: show_scheduled_transactions}}
    show_all_transactions{{link: show_all_transactions}}
    list_transactions --> utter_transactions
    utter_transactions --> list_transactions_next_option
    show_scheduled_transactions --> show_all_transactions
    list_transactions_next_option -->|if slots.list_transactions_next_option == show_scheduled_transactions| show_scheduled_transactions
    list_transactions_next_option -->|if slots.list_transactions_next_option == show_all_transactions| show_all_transactions
    list_transactions_next_option -->|else| utter_no_next_option
    utter_no_next_option --> END
```

## move_money_between_accounts

Transfers money between user's own accounts.

**Flow Guard:** if: False (only triggered via call/link)

```mermaid
graph TD
    END([END])
    self_transfer_source_account[collect: self_transfer_source_account]
    self_transfer_destination_account[collect: self_transfer_destination_account]
    ask_amount[collect: ask_amount]
    check_transfer_funds[check_transfer_funds]
    utter_transfer_money_insufficient_funds[utter_transfer_money_insufficient_funds]
    set_amount_of_money_5[/set_slots<br/>amount_of_money: None/]
    confirm_transfer[collect: confirm_transfer]
    utter_transfer_cancelled[utter_transfer_cancelled]
    execute_transfer[utter_self_transfer_completed]
    self_transfer_source_account --> self_transfer_destination_account
    self_transfer_destination_account --> ask_amount
    ask_amount --> check_transfer_funds
    utter_transfer_money_insufficient_funds --> set_amount_of_money_5
    check_transfer_funds -->|if not slots.has_sufficient_funds| utter_transfer_money_insufficient_funds
    check_transfer_funds -->|else| confirm_transfer
    set_amount_of_money_5 --> ask_amount
    confirm_transfer -->|if not slots.confirmation| utter_transfer_cancelled
    confirm_transfer -->|else| execute_transfer
    utter_transfer_cancelled --> END
    execute_transfer --> END
```

## transfer_money

Handles the following types of money transfers initiated by the user:
- Third Party: Transfer money to another person, business or service within domestic banks. Third party transfer can be immediate, scheduled, or set up as a recurring payments.
- Self transfer: Transfer funds between user's own accounts at the same bank (immediate only).


```mermaid
graph TD
    END([END])
    choose_transfer_type[collect: choose_transfer_type]
    utter_inform_on_wrong_transfer_type[utter_inform_on_wrong_transfer_type]
    perform_self_transfer[[call: perform_self_transfer]]
    perform_third_party_transfer[[call: perform_third_party_transfer]]
    choose_transfer_type -->|if slots.transfer_type == self transfer| perform_self_transfer
    choose_transfer_type -->|if slots.transfer_type == third party| perform_third_party_transfer
    choose_transfer_type -->|else| utter_inform_on_wrong_transfer_type
    utter_inform_on_wrong_transfer_type --> choose_transfer_type
    perform_self_transfer --> END
    perform_third_party_transfer --> END
```

## transfer_money_to_a_third_party

Transferring money to someone else, such as friends, family, or businesses. This
includes paying bills, rent, subscriptions, or transferring money to accounts
outside the user's own portfolio. Third party transfers are limited to recipients
with accounts at domestic banks. Supports immediate, scheduled, and recurring
transfers.


**Flow Guard:** if: False (only triggered via call/link)

```mermaid
graph TD
    END([END])
    recipient_account[collect: recipient_account]
    validate_payee[validate_payee]
    utter_invalid_payee[utter_invalid_payee]
    set_recipient_account_3[/set_slots<br/>recipient_account: None/]
    ask_amount[collect: ask_amount]
    check_transfer_funds[check_transfer_funds]
    utter_transfer_money_insufficient_funds[utter_transfer_money_insufficient_funds]
    set_amount_of_money_7[/set_slots<br/>amount_of_money: None/]
    choose_transfer_timing_type[collect: choose_transfer_timing_type]
    utter_inform_on_wrong_transfer_timing_type[utter_inform_on_wrong_transfer_timing_type]
    execute_immediate_transfer[[call: execute_immediate_transfer]]
    setup_scheduled_transfer[[call: setup_scheduled_transfer]]
    setup_recurring_payment[[call: setup_recurring_payment]]
    recipient_account --> validate_payee
    utter_invalid_payee --> set_recipient_account_3
    ask_amount --> check_transfer_funds
    utter_transfer_money_insufficient_funds --> set_amount_of_money_7
    validate_payee -->|if not slots.is_valid_payee| utter_invalid_payee
    validate_payee -->|else| ask_amount
    set_recipient_account_3 --> END
    check_transfer_funds -->|if not slots.has_sufficient_funds| utter_transfer_money_insufficient_funds
    check_transfer_funds -->|else| choose_transfer_timing_type
    set_amount_of_money_7 --> ask_amount
    choose_transfer_timing_type -->|if slots.transfer_timing_type == immediate| execute_immediate_transfer
    choose_transfer_timing_type -->|if slots.transfer_timing_type == scheduled| setup_scheduled_transfer
    choose_transfer_timing_type -->|if slots.transfer_timing_type == recurring| setup_recurring_payment
    choose_transfer_timing_type -->|else| utter_inform_on_wrong_transfer_timing_type
    utter_inform_on_wrong_transfer_timing_type --> choose_transfer_timing_type
    execute_immediate_transfer --> END
    setup_scheduled_transfer --> END
    setup_recurring_payment --> END
```

## pattern_completed

A flow has been completed and there is nothing else to be done

**Flow Guard:** if: False (only triggered via call/link)

```mermaid
graph TD
```

## pattern_correction

Handle a correction of a slot value.

```mermaid
graph TD
    END([END])
    action_correct_flow_slot[action_correct_flow_slot]
    action_correct_flow_slot --> END
```

## pattern_search

Flow for handling knowledge-based questions

```mermaid
graph TD
    END([END])
    0_action_trigger_search[action_trigger_search]
    0_action_trigger_search --> END
```

## pattern_session_start

Flow for starting the conversation

**nlu_trigger:**
- session_start (confidence_threshold: 0.8)
- test (confidence_threshold: 0.5)

```mermaid
graph TD
    welcome{{link: welcome}}
```

## bill_pay_reminder

Help users set up automatic reminders for recurring bill payments.

```mermaid
graph TD
    END([END])
    biller_name[collect: biller_name]
    bill_due_date[collect: bill_due_date]
    reminder_frequency[collect: reminder_frequency]
    confirm_reminder[collect: confirm_reminder]
    utter_bill_pay_reminder_cancelled[utter_bill_pay_reminder_cancelled]
    setup_bill_pay_reminder[utter_bill_pay_reminder_complete]
    biller_name --> bill_due_date
    bill_due_date --> reminder_frequency
    reminder_frequency --> confirm_reminder
    confirm_reminder -->|if not slots.confirmation| utter_bill_pay_reminder_cancelled
    confirm_reminder -->|else| setup_bill_pay_reminder
    utter_bill_pay_reminder_cancelled --> END
    setup_bill_pay_reminder --> END
```

## check_balance

Guides the user through retrieving their account balance by requesting an
account number or an account name and returning the balance on the selected account


```mermaid
graph TD
    check_balance[check_balance]
    utter_current_balance[utter_current_balance]
    check_balance --> utter_current_balance
```

## download_statements

Allow users to download their monthly account statements in PDF or CSV format.

```mermaid
graph TD
    END([END])
    statement_month[collect: statement_month]
    statement_year[collect: statement_year]
    statement_format[collect: statement_format]
    confirmation[collect: confirmation]
    utter_statement_download_cancelled[utter_statement_download_cancelled]
    initiate_download[utter_statement_download_complete]
    statement_month --> statement_year
    statement_year --> statement_format
    statement_format --> confirmation
    confirmation -->|if not slots.confirmation| utter_statement_download_cancelled
    confirmation -->|else| initiate_download
    utter_statement_download_cancelled --> END
    initiate_download --> END
```

