# CALM routing evaluation export

This folder contains a batch runner for **Rasa Pro 3.16** CALM assistants.

It sends user utterances through your trained assistant and writes a CSV with the routing signals needed to compute metrics such as Recall@K offline.

The runner does **not** calculate Recall@K or any other metric — it only exports the raw predictions and runtime outcomes.

---

## What you get

For every input utterance the output CSV includes:

| Column | Meaning |
| --- | --- |
| `test_id` | Your identifier for this test row |
| `sender_id` | Rasa runtime session key used for `handle_message` (auto-generated if not provided) |
| `user_utterance` | The message sent to the assistant |
| `expected_flow_id` | Optional ground truth you provide in the input CSV (echoed back unchanged) |
| `predicted_commands` | Commands predicted by the LLM command generator (JSON) |
| `flows_from_semantic_search` | Flow-retrieval candidates as `[{"flow_id": ..., "score": ...}, ...]` (empty array if flow retrieval is disabled) |
| `flows_in_prompt` | Flow IDs passed as context to the LLM prompt (JSON array of strings) |
| `runtime_flow_started_ids` | Flows that actually started at runtime (comma-separated; excludes `pattern_clarification`) |
| `runtime_clarification_ids` | Flow options offered when `pattern_clarification` triggered (comma-separated) |
| `latency_seconds` | End-to-end time for that message |
| `error` | Error message if processing failed for that row |

### Which columns to use for Recall@K

| Goal | Column(s) to use |
| --- | --- |
| Retrieval quality | `flows_from_semantic_search` (ordered by score, highest first) |
| Prompt coverage | `flows_in_prompt` |
| Clarification quality | `runtime_clarification_ids` |
| Final routing decision | `runtime_flow_started_ids` |

Compare any of these against your `expected_flow_id` in your own spreadsheet, notebook, or BI tool.

---

## Prerequisites

1. **Rasa Pro 3.16** installed in your environment
2. A **trained CALM model** (`rasa train`)
3. Required model/API credentials configured
4. Custom actions available if your flows call them:
   - Run `rasa run actions` in another terminal, **or**
   - Use [stubbed custom actions](https://rasa.com/docs/rasa-pro/production/testing-your-assistant#stubbing-custom-actions) in `endpoints.yml` for faster batch runs

---

## Input format

Provide a CSV with at least these columns:

| Column | Required | Description |
| --- | --- | --- |
| `test_id` | Yes | Unique ID for the row |
| `user_utterance` | Yes | Message to send |
| `sender_id` | No | Rasa runtime session key. If omitted, a UUID is generated per row |
| `expected_flow_id` | No | Your ground-truth flow ID. Not validated by the runner |

See `evaluation/examples/sample_input.csv` for an example and
`evaluation/examples/sample_output.csv` for the corresponding output shape.

> **Note:** `sample_output.csv` is illustrative. Run the script against your trained model to generate real results.

Each row in the input is evaluated independently with a single `handle_message` call — there is no conversation state carried between rows.

---

## Run the export

```bash
python evaluation/run_routing_eval.py \
  --model models/ \
  --endpoints endpoints.yml \
  --input evaluation/examples/sample_input.csv \
  --output evaluation/results/routing_eval.csv
```

Arguments:

| Flag | Description |
| --- | --- |
| `--model` | Path to trained model directory or `.tar.gz` archive |
| `--endpoints` | Path to `endpoints.yml` (default: `endpoints.yml`) |
| `--input` | Input CSV path |
| `--output` | Output CSV path (parent folders are created automatically) |
| `--remote-storage` | Optional remote model storage backend |

---

## Example workflow

```bash
# 1. Train the assistant
rasa train

# 2. (Optional) Start custom actions
rasa run actions

# 3. Export routing signals
python evaluation/run_routing_eval.py \
  --model models/ \
  --input my_test_set.csv \
  --output my_results.csv

# 4. Compute Recall@K in your own tooling
```

---

## Notes for large datasets (~100k utterances)

- Watch LLM API rate limits — consider adding retry/back-off logic for large runs
- Use stubbed custom actions to avoid action-server overhead when you only care about routing
- Split large input CSVs into batches and concatenate outputs afterward

---

## Troubleshooting

| Problem | Likely cause |
| --- | --- |
| `Agent is not ready` | Model path is wrong or model failed to load |
| `only supports CALM assistants` | Assistant is not CALM-based |
| Empty `flows_from_semantic_search` | Flow retrieval is not enabled in `config.yml` |
| Empty `flows_in_prompt` | No flow context was added to the prompt for that turn |
| Rows with `error` populated | Missing action server, API failure, or runtime exception |

For more on CALM routing, see the [Dialogue Understanding docs](https://rasa.com/docs/rasa-pro/concepts/dialogue-understanding).
