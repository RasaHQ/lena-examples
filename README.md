# Finance Banking Agent Template

A lightweight banking conversational agent template for the fictional **Fenlo Bank** that handles account management, card services, and money transfers.

## 🚀 What's Included

This template provides a banking assistant with:

- **Card Services**: Card activation, blocking, replacement, and listing  
- **Money Transfers**: Account-to-account transfers and third-party payments
- **Contact Management**: Add, list, and remove trusted contacts
- **Banking Knowledge**: FAQ system with Fenlo Bank documentation
- **NLU flow triggers with low-confidence fallback**: Card and transfer flows can be started from classified intents; when confidence is below the flow’s trigger threshold, a dedicated fallback flow asks the user to rephrase (see [Testing the Fallback NLU Command Adapter](#testing-the-fallback-nlu-command-adapter) below).

## Testing the Fallback NLU Command Adapter

This template includes a custom **`FallbackNLUCommandAdapter`** that extends Rasa’s `NLUCommandAdapter`. When the classifier picks an intent that matches a flow’s NLU trigger **but** confidence is **below** that trigger’s `confidence_threshold`, the assistant starts a small **fallback** flow instead of the main skill flow, and asks the user to rephrase.

**Full behavior, configuration, and caveats** are documented here:

**[custom/fallback_nlu_command_adapter.md](custom/fallback_nlu_command_adapter.md)**

### Prerequisites

- Install dependencies into the same environment you use for the Rasa CLI, for example:

  ```bash
  pip install -r requirements.txt
  ```

### How to try it (Rasa Inspector)

1. From the project root, train the latest model: `rasa train --data data --domain domain` (see [`rasa train`](https://rasa.com/docs/reference/api/command-line-interface#rasa-train)).
2. Run the inspector: `rasa inspect --debug` (see [`rasa inspect`](https://rasa.com/docs/reference/api/command-line-interface#rasa-inspect) and [Trying your assistant / Rasa Inspector](https://rasa.com/docs/pro/testing/trying-assistant/)).
3. Send test messages in the inspector and watch **which flow starts** (for example `flow_started` on the target skill vs. the configured **`fallback`** rephrase flow).

**What to look for**

- **Normal path:** confidence is at or above the trigger threshold → the **intended** flow starts (for example `block_card`, `list_cards`).
- **Fallback path:** same top intent as a trigger, but confidence **below** that flow’s threshold → the configured fallback flow starts (in this repo the flow id is **`fallback`**, with a short “please rephrase” style response).

Exact confidence scores depend on your NLU pipeline and training data, so borderline phrases are **model-dependent**.

### Example phrases to try with a client

Use these as a **starting script**; refine them after you inspect real confidences in **Rasa Inspector** on your build.

| You say (example) | What we are illustrating |
| ------------------- | ------------------------- |
| `show my cards` | Clear **list cards** wording (often starts the list-cards flow when confidence is high). |
| `block my card` | Clear **block card** wording (often starts the block-card flow when confidence is high). |
| `what about my cards` | More vague “cards” wording—**may** stay on `list_cards` intent but with **lower** confidence vs. a crisp phrase; useful for exploring the fallback path for that flow. |
| `something about blocking cards` | Vague **block**-adjacent wording from training—useful for exploring borderline behavior. |
| `I want to transfer money` | Clear **transfer money** entry (when that flow and NLU data are present). |

If the bot answers with the **generic rephrase** message instead of jumping straight into slot collection for the skill, you are likely seeing the **low-confidence fallback** path.

## 📁 Directory Structure

```
├── actions/         # Custom Python logic for banking operations
├── custom/          # Custom Rasa components (e.g. Fallback NLU command adapter)
├── data/            # Banking conversation flows and training data
├── domain/          # Banking agent configuration
├── db/              # Mock JSON database for testing
├── docs/            # Fenlo Bank knowledge base and FAQ documents
├── prompts/         # LLM prompts for enhanced banking responses
├── requirements.txt # Python dependencies (install before `rasa train` / `rasa inspect`)
└── config.yml       # Training pipeline configuration
```
