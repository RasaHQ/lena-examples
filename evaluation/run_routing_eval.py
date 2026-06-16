#!/usr/bin/env python3
"""Batch routing evaluation runner for CALM assistants (Rasa Pro 3.16+).

Reads a CSV of user utterances, sends each one through a trained CALM assistant
via `agent.handle_message`, and writes the routing signals to an output CSV.

Usage:
    python evaluation/run_routing_eval.py \
        --model models/ \
        --input evaluation/examples/sample_input.csv \
        --output results/routing_eval.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import structlog

from rasa.core.agent import Agent, load_agent
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.exceptions import AgentNotReady
from rasa.dialogue_understanding.patterns.clarify import FLOW_PATTERN_CLARIFICATION
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.shared.agents.agent_setup import AgentsConnectionCleanup
from rasa.shared.core.events import Event, FlowStarted, UserUttered
from rasa.shared.nlu.constants import (
    FLOWS_FROM_SEMANTIC_SEARCH,
    FLOWS_IN_PROMPT,
    PREDICTED_COMMANDS,
)

structlogger = structlog.get_logger()

# ── Input ──────────────────────────────────────────────────────────────────────
# Required columns in the input CSV.
INPUT_REQUIRED_COLUMNS = ("test_id", "user_utterance")

# ── Output ─────────────────────────────────────────────────────────────────────
# Columns written to the output CSV.
OUTPUT_COLUMNS = (
    "test_id",
    "sender_id",
    "user_utterance",
    "expected_flow_id",
    # What the LLM command generator predicted
    "predicted_commands",
    # Flows ranked by the flow retrieval step (empty if flow retrieval is off)
    "flows_from_semantic_search",
    # Flows passed as context to the LLM prompt
    "flows_in_prompt",
    # What actually happened at runtime
    "runtime_flow_started_ids",
    "runtime_clarification_ids",
    # Diagnostics
    "latency_seconds",
    "error",
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CSV of user utterances through a trained CALM assistant "
            "and export routing signals for offline metric calculation."
        )
    )
    parser.add_argument("--model", required=True, help="Path to trained model.")
    parser.add_argument(
        "--endpoints",
        default="endpoints.yml",
        help="Path to endpoints.yml (default: endpoints.yml).",
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


# ── Input parsing ──────────────────────────────────────────────────────────────

def read_rows(path: Path) -> List[Dict[str, Any]]:
    """Read and validate the input CSV."""
    rows: List[Dict[str, Any]] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {path}")

        missing = [
            col for col in INPUT_REQUIRED_COLUMNS
            if col not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"Input CSV must include columns {INPUT_REQUIRED_COLUMNS}. "
                f"Missing: {missing}"
            )

        for line_number, record in enumerate(reader, start=2):
            utterance = (record.get("user_utterance") or "").strip()
            if not utterance:
                structlogger.warning(
                    "routing_eval.skip_empty_utterance",
                    line=line_number,
                    test_id=record.get("test_id"),
                )
                continue

            test_id = (record.get("test_id") or "").strip() or f"row_{line_number}"

            # sender_id is the Rasa runtime tracker/session key.
            # If not provided, generate a fresh UUID so each row is isolated.
            sender_id = (record.get("sender_id") or "").strip() or str(uuid.uuid4())

            rows.append(
                {
                    "test_id": test_id,
                    "sender_id": sender_id,
                    "user_utterance": utterance,
                    "expected_flow_id": (record.get("expected_flow_id") or "").strip(),
                }
            )

    if not rows:
        raise ValueError(f"No usable rows found in {path}")

    return rows


# ── Agent setup ────────────────────────────────────────────────────────────────

async def setup_agent(model_path: str, endpoints_path: Path) -> Agent:
    """Load and validate the trained CALM agent."""
    endpoints = AvailableEndpoints.read_endpoints(endpoints_path)

    async with AgentsConnectionCleanup():
        agent = await load_agent(
            model_path=model_path,
            endpoints=endpoints,
        )

    if not agent.is_ready():
        raise AgentNotReady("Agent is not ready.")
    if agent.processor is None:
        raise AgentNotReady("Agent processor is missing.")
    if not agent.processor.is_calm_assistant:
        raise AgentNotReady("This runner supports CALM assistants only.")

    return agent


# ── Signal extraction ──────────────────────────────────────────────────────────

def _extract_signals(events: List[Event]) -> Dict[str, Any]:
    """Pull routing signals out of the tracker events for the latest user turn."""
    # Find the latest UserUttered event — that is where Rasa stores parse_data
    # including predicted_commands, flows_from_semantic_search, flows_in_prompt.
    latest_user_idx = None
    for idx, event in enumerate(events):
        if isinstance(event, UserUttered):
            latest_user_idx = idx

    if latest_user_idx is None:
        return _empty_signals("No UserUttered event found in tracker.")

    user_event = events[latest_user_idx]
    if not isinstance(user_event, UserUttered):
        return _empty_signals("Latest event is not UserUttered.")

    parse_data = user_event.parse_data or {}
    turn_events = events[latest_user_idx + 1:]

    # Collect which flows actually started vs triggered clarification at runtime.
    runtime_flow_started_ids: List[str] = []
    runtime_clarification_ids: List[str] = []
    for event in turn_events:
        if not isinstance(event, FlowStarted):
            continue
        if event.flow_id == FLOW_PATTERN_CLARIFICATION:
            runtime_clarification_ids.extend(
                event.metadata.get("clarification_ids", []) or []
            )
        else:
            runtime_flow_started_ids.append(event.flow_id)

    # Format flow retrieval candidates as [{flow_id, score}, ...].
    retrieval_candidates = parse_data.get(FLOWS_FROM_SEMANTIC_SEARCH) or []
    retrieval_json = [
        {"flow_id": str(item[0]), "score": float(item[1])}
        for item in retrieval_candidates
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]

    return {
        "predicted_commands": json.dumps(
            parse_data.get(PREDICTED_COMMANDS) or {}, sort_keys=True
        ),
        "flows_from_semantic_search": json.dumps(retrieval_json, sort_keys=True),
        "flows_in_prompt": json.dumps(
            [str(v) for v in (parse_data.get(FLOWS_IN_PROMPT) or [])],
            sort_keys=True,
        ),
        "runtime_flow_started_ids": ",".join(dict.fromkeys(runtime_flow_started_ids)),
        "runtime_clarification_ids": ",".join(dict.fromkeys(runtime_clarification_ids)),
        "error": "",
    }


def _empty_signals(error: str) -> Dict[str, Any]:
    return {
        "predicted_commands": "{}",
        "flows_from_semantic_search": "[]",
        "flows_in_prompt": "[]",
        "runtime_flow_started_ids": "",
        "runtime_clarification_ids": "",
        "error": error,
    }


# ── Per-row evaluation ─────────────────────────────────────────────────────────

async def evaluate_row(agent: Agent, row: Dict[str, Any]) -> Dict[str, Any]:
    """Send one utterance through the agent and return a result row."""
    sender_id = row["sender_id"]
    output_channel = CollectingOutputChannel()

    if agent.processor is None:
        raise AgentNotReady("Agent processor is missing.")

    # Start a fresh session for this sender_id.
    await agent.processor.fetch_tracker_with_initial_session(
        sender_id, output_channel=output_channel
    )

    start = time.time()
    error_message = ""
    try:
        # set_record_commands_and_prompts() ensures predicted_commands,
        # flows_from_semantic_search and flows_in_prompt are written to parse_data.
        with set_record_commands_and_prompts():
            await agent.handle_message(
                UserMessage(row["user_utterance"], output_channel, sender_id)
            )
    except Exception as exc:
        error_message = str(exc)

    tracker = await agent.tracker_store.retrieve(sender_id)
    signals = (
        _extract_signals(list(tracker.events or []))
        if tracker is not None
        else _empty_signals("Tracker retrieval failed.")
    )

    return {
        "test_id": row["test_id"],
        "sender_id": sender_id,
        "user_utterance": row["user_utterance"],
        "expected_flow_id": row["expected_flow_id"],
        "predicted_commands": signals["predicted_commands"],
        "flows_from_semantic_search": signals["flows_from_semantic_search"],
        "flows_in_prompt": signals["flows_in_prompt"],
        "runtime_flow_started_ids": signals["runtime_flow_started_ids"],
        "runtime_clarification_ids": signals["runtime_clarification_ids"],
        "latency_seconds": f"{time.time() - start:.4f}",
        "error": error_message or signals["error"],
    }


# ── Output ─────────────────────────────────────────────────────────────────────

def write_output(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ── Entry point ────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    input_rows = read_rows(Path(args.input))
    agent = await setup_agent(args.model, Path(args.endpoints))

    results = [await evaluate_row(agent, row) for row in input_rows]

    write_output(Path(args.output), results)
    structlogger.info(
        "routing_eval.complete",
        rows=len(results),
        errors=sum(1 for r in results if r["error"]),
        output=args.output,
    )


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except AgentNotReady as exc:
        structlogger.error("routing_eval.agent_not_ready", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
