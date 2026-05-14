# Low Confidence Fallback NLU Command Adapter

This custom component extends Rasa's `NLUCommandAdapter` and starts a configured
fallback flow when a message is a **near-miss**:

- intent name matches a flow trigger intent, but
- confidence is below that trigger's `confidence_threshold`.

It is useful when you want a flow-level clarification path without rewriting the
intent to `nlu_fallback`.

## File

- Component: [`custom/fallback_nlu_command_adapter.py`](fallback_nlu_command_adapter.py)
- Class: `FallbackNLUCommandAdapter`

## How to Configure

In your assistant `config.yml`, replace `NLUCommandAdapter` with this class (or use it instead of the stock adapter in your NLU pipeline):

```yaml
pipeline:
  - name: custom.fallback_nlu_command_adapter.FallbackNLUCommandAdapter
    fallback_flow_id: card_nlu_fallback
```

If this component appears in the pipeline, low-confidence fallback behavior is **on**. To disable it, remove the component from `config.yml` (or swap in the stock `NLUCommandAdapter`).

### Parameters

- `fallback_flow_id` (required): Flow id to start on near-miss.

## How It Works

1. Runs normal `NLUCommandAdapter` logic via `super().predict_commands(...)`.
2. If commands already contain `StartFlowCommand` or `ErrorCommand`, it leaves
   behavior unchanged.
3. Otherwise, it scans all non-default user flows and checks trigger conditions:
   same intent name + confidence below threshold.
4. If matched, it **replaces** the current command list with `StartFlowCommand(fallback_flow_id)` (and optionally `SetSlotCommand` for coexistence routing).
5. If coexistence routing is active, it also sets `ROUTE_TO_CALM_SLOT`.
6. Applies `clean_up_commands(...)` before returning.

## Requirements and Caveats

- `fallback_flow_id` must exist in the loaded flows.
- The fallback flow must be startable in the current context (`if` guards still
  apply in the command pipeline).
- To avoid loops, the component skips evaluating the fallback flow itself as a
  near-miss candidate.
- This component does not modify `FallbackClassifier`; global fallback behavior
  stays the same.

## Quick Validation

After configuring:

1. Send a message with intent confidence below a target flow threshold but above
   your global fallback threshold.
2. Confirm tracker events show `flow_started` for `fallback_flow_id`.
3. Send a clear phrase above threshold and confirm original flow starts as usual.

## Example Scenario

- Flow trigger:
  - `intent: Cartes__limites__augmentation`
  - `confidence_threshold: 0.7`
- User message classified as:
  - intent `Cartes__limites__augmentation`
  - confidence `0.54`

Result:

- Original flow does not start (below 0.7).
- This component starts `fallback_flow_id` (for example, `card_nlu_fallback`).
