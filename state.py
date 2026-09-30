"""
state.py — Shared state for the Flight Disruption Rebooking Assistant.

Design decisions:
- `messages` uses the `add_messages` reducer so every node appends rather than replaces.
- All other fields are plain values; each node updates only the fields it owns.
- `retry_count` starts at 0 and is incremented by `policy_checker` on each failure.
- `escalation_status` becomes True when escalation_agent runs.
- `active_agent` is set by each specialist so the `tools` node can route back correctly.
"""
from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class RebookingState(TypedDict):
    # ── Conversation history ──────────────────────────────────────────
    # add_messages reducer: new messages are appended, not overwritten.
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Request intake ────────────────────────────────────────────────
    request_id: str
    booking_ref: str

    # ── Classifier outputs ────────────────────────────────────────────
    intent: Optional[str]           # rebook | refund | compensation | complaint
    constraints: Optional[str]     # e.g. "must arrive by noon tomorrow"

    # ── Data gathered via tools ───────────────────────────────────────
    booking_details: Optional[dict]

    # ── Specialist agent output ───────────────────────────────────────
    proposed_solution: Optional[dict]   # {type, ...fields}

    # ── Policy checker ────────────────────────────────────────────────
    policy_check_result: Optional[str]  # "pass" | "fail"
    failure_reason: Optional[str]       # human-readable rule that was broken
    retry_count: int                    # 0 on first attempt; max 2 retries before escalation

    # ── Escalation & final output ─────────────────────────────────────
    escalation_status: bool             # True once escalation_agent has run
    escalation_note: Optional[str]      # Handover text written by escalation_agent
    final_response: Optional[str]       # Reply sent to the passenger

    # ── Routing helper ────────────────────────────────────────────────
    active_agent: Optional[str]         # Which specialist is currently handling the request
