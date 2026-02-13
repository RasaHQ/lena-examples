# Finance Banking Agent Template

A lightweight banking conversational agent template for the fictional **Fenlo Bank** that handles account management, card services, and money transfers.

## 🚀 What's Included

This template provides a banking assistant with:
- **Account Management**: Balance checking and statement downloads
- **Card Services**: Card activation, blocking, replacement, and listing  
- **Money Transfers**: Account-to-account transfers and third-party payments
- **Contact Management**: Add, list, and remove trusted contacts
- **Bill Management**: Bill payment reminders and scheduling
- **Banking Knowledge**: FAQ system with Fenlo Bank documentation

## 📁 Directory Structure

```
├── actions/         # Custom Python logic for banking operations
├── data/            # Banking conversation flows and training data
├── domain/          # Banking agent configuration
├── db/              # Mock JSON database for testing
├── docs/            # Fenlo Bank knowledge base and FAQ documents
├── prompts/         # LLM prompts for enhanced banking responses
└── config.yml       # Training pipeline configuration
```

## 🔧 Visualizing Flows

Use `visualize_flows.py` to convert YAML flow definitions into Mermaid diagrams for better visualization:

```bash
# Visualize a single flow file
python visualize_flows.py data/transfers/transfer_money.yml

# Visualize all flows in a directory
python visualize_flows.py data/

# Specify custom output path
python visualize_flows.py data/ output/flows.md
```

The script generates markdown files with Mermaid diagrams showing:
- Flow steps and their connections
- Decision branches and conditions
- Slot collection and manipulation
- Flow transitions (link/call operations)

**Output:** By default, outputs to `flows.md` in the input directory.
