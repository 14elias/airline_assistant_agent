"""
nodes.py — All LangGraph node functions for the Rebooking Assistant.

Each node takes RebookingState and returns a partial state update dict.
Nodes only write the fields they own.

Key design:
- Specialist agents (rebooking/refund/compensation) run a tool-calling loop via ToolNode.
- When the model stops calling tools, the agent builds the proposal DETERMINISTICALLY
  from the raw tool results using Python — no second LLM call, works with any model.
- policy_checker is pure Python (no LLM).
- escalation_agent and final_response_agent use plain LLM text generation.

LLM: Groq (openai/gpt-oss-120b) via OpenAI-compatible endpoint.
"""
import os
import re
import json
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from state import RebookingState
from tools import get_booking, search_flights, get_fare_rules
from prompts import (
    CLASSIFIER_PROMPT,
    CLARIFY_CONSTRAINTS_MSG,
    REBOOKING_AGENT_PROMPT,
    REFUND_AGENT_PROMPT,
    COMPENSATION_AGENT_PROMPT,
    ESCALATION_AGENT_PROMPT,
    FINAL_RESPONSE_ESCALATED_PROMPT,
    FINAL_RESPONSE_RESOLVED_PROMPT,
)

# ── LLM configuration ─────────────────────────────────────────────────────────
# Groq API is OpenAI-compatible; set GROQ_API_KEY in your .env file.
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY", "not-set"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)

# ── Pydantic schema for classifier (structured output) ────────────────────────
from pydantic import BaseModel, Field

class ClassifierOutput(BaseModel):
    intent: str = Field(
        description="Passenger intent. Must be exactly one of: rebook, refund, compensation, complaint"
    )
    constraints: str = Field(
        description="Any specific constraints the passenger mentioned (e.g. 'must arrive by noon'). "
                    "Write 'none' if there are no constraints."
    )


# ── Helper: extract ToolMessage results from message history ──────────────────

def _get_tool_result(messages: list, tool_name: str) -> Optional[dict]:
    """Find the most recent ToolMessage from a named tool and parse its JSON content."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and "error" not in data:
                    return data
                if isinstance(data, list):
                    return {"results": data}
            except Exception:
                pass
    return None


def _get_tool_results_list(messages: list, tool_name: str) -> list:
    """Find the most recent ToolMessage from a named tool and return as list."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            try:
                data = json.loads(msg.content)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data]
            except Exception:
                pass
    return []


# ── Node 1: Request Classifier ────────────────────────────────────────────────

def request_classifier(state: RebookingState) -> dict:
    """
    Uses structured output to extract intent and constraints from the passenger message.
    Writes: intent, constraints.
    """
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    sys_msg = SystemMessage(content=CLASSIFIER_PROMPT)

    structured_llm = llm.with_structured_output(ClassifierOutput)
    result: ClassifierOutput = structured_llm.invoke([sys_msg, HumanMessage(content=last_human)])

    return {
        "intent": result.intent.lower().strip(),
        "constraints": result.constraints,
    }


# ── Node 1b: Clarify Constraints (Bonus) ─────────────────────────────────────

# Phrases that signal the passenger gave no real constraints
_VAGUE_CONSTRAINTS = {"none", "no constraints", "n/a", "na", "", "not mentioned", "no specific constraints"}

def clarify_constraints(state: RebookingState) -> dict:
    """
    Bonus node — only reached for 'rebook' intent.
    If the passenger's constraints are vague (e.g. 'none'), pause with interrupt()
    and ask them for details before flight search begins.
    The human's answer is injected back as the new constraints value.

    Writes: constraints (updated if clarification was needed).
    """
    constraints = (state.get("constraints") or "none").strip().lower()

    if constraints in _VAGUE_CONSTRAINTS:
        # Pause the graph and ask the passenger
        answer = interrupt(CLARIFY_CONSTRAINTS_MSG)
        # `answer` is whatever the human typed when resuming the graph
        return {"constraints": str(answer).strip()}

    # Constraints are clear — continue without interrupting
    return {}


# ── Node 2: Rebooking Agent ───────────────────────────────────────────────────

def rebooking_agent(state: RebookingState) -> dict:
    """
    Tool-calling agent. Loops with ToolNode to get booking + search flights.
    Once tools are done, builds the proposal deterministically from tool results.
    Writes: messages, active_agent, proposed_solution, booking_details.
    """
    failure_note = ""
    if state.get("failure_reason"):
        failure_note = (
            f"\n\nIMPORTANT: Your previous proposal was REJECTED. Reason: {state['failure_reason']}. "
            "The only available flight (AF100 Business class) will always fail the cabin policy. "
            "Stop trying — there are no valid flights."
        )

    sys_msg = SystemMessage(
        content=REBOOKING_AGENT_PROMPT.format(
            booking_ref=state.get("booking_ref", "N/A"),
            constraints=state.get("constraints", "none"),
            failure_note=failure_note
        )
    )

    agent_llm = llm.bind_tools([get_booking, search_flights])
    all_messages = [sys_msg] + state["messages"]
    response: AIMessage = agent_llm.invoke(all_messages)

    updates: dict = {"messages": [response], "active_agent": "rebooking_agent"}

    # Still has tool calls → let ToolNode handle them (tools_condition routes there)
    if response.tool_calls:
        return updates

    # No more tool calls → build proposal from ALL tool results in history
    # Must include the full state["messages"] as ToolMessages may be from prior loop iterations
    all_msgs = list(state["messages"]) + [response]
    booking = _get_tool_result(all_msgs, "get_booking")
    flights = _get_tool_results_list(all_msgs, "search_flights")

    if booking:
        updates["booking_details"] = booking

    if flights and isinstance(flights[0], dict) and "error" not in flights[0]:
        flight = flights[0]  # pick first available flight
        updates["proposed_solution"] = {
            "type": "rebook",
            "flight_no": flight.get("flight_no", "UNKNOWN"),
            "cabin": flight.get("cabin", "Economy"),
            "connection_time": int(flight.get("connection_time", 0)),
            "departs_in_hours": int(flight.get("departs_in_hours", 0)),
        }
    else:
        updates["proposed_solution"] = {
            "type": "rebook",
            "error": "No valid flights found in search results.",
        }

    return updates


# ── Node 3: Refund Agent ──────────────────────────────────────────────────────

def refund_agent(state: RebookingState) -> dict:
    """
    Tool-calling agent. Gets booking + fare rules, then proposes refund from data.
    Writes: messages, active_agent, proposed_solution, booking_details.
    """
    sys_msg = SystemMessage(
        content=REFUND_AGENT_PROMPT.format(
            booking_ref=state.get("booking_ref", "N/A")
        )
    )

    agent_llm = llm.bind_tools([get_booking, get_fare_rules])
    response: AIMessage = agent_llm.invoke([sys_msg] + state["messages"])

    updates: dict = {"messages": [response], "active_agent": "refund_agent"}

    if response.tool_calls:
        return updates

    all_msgs = state["messages"] + [response]
    booking = _get_tool_result(all_msgs, "get_booking")
    fare_rules = _get_tool_result(all_msgs, "get_fare_rules")

    if booking:
        updates["booking_details"] = booking

    if booking:
        disruption = booking.get("disruption_details", "").lower()
        fare_type = booking.get("fare_type", "non-refundable")

        if "cancelled" in disruption:
            # Airline cancelled → full fare refund
            amount_description = "full fare"
            amount_usd = 500.0  # Mock full fare amount
        elif fare_type == "non-refundable":
            # Passenger-initiated on non-refundable → taxes only
            tax = fare_rules.get("tax_amount", 75) if fare_rules else 75
            amount_description = "taxes only"
            amount_usd = float(tax)
        else:
            # Refundable fare, no cancellation → full fare
            amount_description = "full fare"
            amount_usd = 500.0

        updates["proposed_solution"] = {
            "type": "refund",
            "amount_description": amount_description,
            "amount_usd": amount_usd,
        }
    else:
        updates["proposed_solution"] = {"type": "refund", "error": "Could not retrieve booking."}

    return updates


# ── Node 4: Compensation Agent ────────────────────────────────────────────────

def compensation_agent(state: RebookingState) -> dict:
    """
    Tool-calling agent. Gets booking disruption details, proposes compensation by policy.
    Writes: messages, active_agent, proposed_solution, booking_details.
    """
    sys_msg = SystemMessage(
        content=COMPENSATION_AGENT_PROMPT.format(
            booking_ref=state.get("booking_ref", "N/A")
        )
    )

    agent_llm = llm.bind_tools([get_booking])
    response: AIMessage = agent_llm.invoke([sys_msg] + state["messages"])

    updates: dict = {"messages": [response], "active_agent": "compensation_agent"}

    if response.tool_calls:
        return updates

    all_msgs = state["messages"] + [response]
    booking = _get_tool_result(all_msgs, "get_booking")

    if booking:
        updates["booking_details"] = booking

    if booking:
        disruption = booking.get("disruption_details", "").lower()

        # Apply policy table deterministically
        if "cancelled" in disruption:
            amount_usd = 400.0
            reason = "Flight cancelled by airline → $400 compensation"
        elif "delay" in disruption:
            match = re.search(r"(\d+)\s*hour", disruption)
            hours = int(match.group(1)) if match else 0
            if hours < 3:
                amount_usd = 0.0
                reason = f"{hours}-hour delay → $0 (under 3 hours)"
            elif hours <= 6:
                amount_usd = 200.0
                reason = f"{hours}-hour delay → $200 (3–6 hours)"
            else:
                amount_usd = 400.0
                reason = f"{hours}-hour delay → $400 (over 6 hours)"
        else:
            amount_usd = 0.0
            reason = "No qualifying disruption found"

        updates["proposed_solution"] = {
            "type": "compensation",
            "amount_usd": amount_usd,
            "reason": reason,
        }
    else:
        updates["proposed_solution"] = {"type": "compensation", "error": "Could not retrieve booking."}

    return updates


# ── Node 5: Policy Checker ────────────────────────────────────────────────────

def policy_checker(state: RebookingState) -> dict:
    """
    Pure Python node — no LLM.
    Validates proposed_solution against airline policy rules.
    On failure, stores the reason and increments retry_count.
    Writes: policy_check_result, failure_reason, retry_count, booking_details.
    """
    sol = state.get("proposed_solution") or {}
    sol_type = sol.get("type", "")

    # Guard: no proposal or error
    if not sol or "error" in sol or not sol_type:
        return {
            "policy_check_result": "fail",
            "failure_reason": f"No valid solution was proposed: {sol.get('error', 'unknown error')}",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # Get booking details (from state cache or fetch directly)
    booking = state.get("booking_details") or {}
    if not booking:
        booking = get_booking.invoke({"booking_ref": state.get("booking_ref", "")})
        if "error" in booking:
            booking = {}

    is_valid = True
    reason = ""

    # ── Rebook rules ──────────────────────────────────────────────────────────
    if sol_type == "rebook":
        cabin_rank = {"Economy": 1, "Premium Economy": 2, "Business": 3, "First": 4}
        orig_cabin = booking.get("cabin_class", "Economy")
        new_cabin = sol.get("cabin", "Economy")

        if cabin_rank.get(new_cabin, 1) > cabin_rank.get(orig_cabin, 1):
            is_valid = False
            reason = (
                f"Cabin upgrade not allowed: proposed '{new_cabin}' is higher than "
                f"original '{orig_cabin}'."
            )
        elif int(sol.get("connection_time", 0)) < 60:
            is_valid = False
            reason = f"Connection time {sol.get('connection_time')} min is below the 60-minute minimum."
        elif int(sol.get("departs_in_hours", 0)) > 48:
            is_valid = False
            reason = "Flight departs more than 48 hours from now."

    # ── Refund rules ──────────────────────────────────────────────────────────
    elif sol_type == "refund":
        disruption = booking.get("disruption_details", "").lower()
        fare_type = booking.get("fare_type", "non-refundable")
        amt_desc = sol.get("amount_description", "").lower()

        if "cancelled" in disruption and "full" not in amt_desc:
            is_valid = False
            reason = "Airline cancelled: passenger is entitled to a full fare refund."
        elif "cancelled" not in disruption and fare_type == "non-refundable" and "tax" not in amt_desc:
            is_valid = False
            reason = "Non-refundable fare with passenger-initiated change: taxes only."

        # Bonus: Human-in-the-loop for refund > $300
        if is_valid:
            try:
                if float(sol.get("amount_usd", 0)) > 300:
                    answer = interrupt(
                        f"⚠️ Human approval required: refund of ${sol['amount_usd']:.2f} exceeds $300. "
                        "Approve or reject."
                    )
                    if answer and "reject" in str(answer).lower():
                        is_valid = False
                        reason = f"Human supervisor rejected the refund. Note: {answer}"
            except (ValueError, TypeError):
                pass

    # ── Compensation rules ────────────────────────────────────────────────────
    elif sol_type == "compensation":
        disruption = booking.get("disruption_details", "").lower()
        proposed_amt = float(sol.get("amount_usd", -1))

        if "cancelled" in disruption:
            expected_amt = 400.0
        elif "delay" in disruption:
            match = re.search(r"(\d+)\s*hour", disruption)
            hours = int(match.group(1)) if match else 0
            if hours < 3:
                expected_amt = 0.0
            elif hours <= 6:
                expected_amt = 200.0
            else:
                expected_amt = 400.0
        else:
            expected_amt = 0.0

        if proposed_amt != expected_amt:
            is_valid = False
            reason = (
                f"Compensation ${proposed_amt:.0f} is incorrect. "
                f"Policy requires ${expected_amt:.0f} for this disruption."
            )

        # Bonus: Human-in-the-loop for compensation > $300
        if is_valid and proposed_amt > 300:
            answer = interrupt(
                f"⚠️ Human approval required: compensation of ${proposed_amt:.2f} exceeds $300. "
                "Approve or reject."
            )
            if answer and "reject" in str(answer).lower():
                is_valid = False
                reason = f"Human supervisor rejected the compensation. Note: {answer}"

    retry_count = state.get("retry_count", 0)
    if not is_valid:
        retry_count += 1

    return {
        "policy_check_result": "pass" if is_valid else "fail",
        "failure_reason": reason if not is_valid else "",
        "retry_count": retry_count,
        "booking_details": booking,
    }


# ── Node 6: Escalation Agent ──────────────────────────────────────────────────

def escalation_agent(state: RebookingState) -> dict:
    """
    Writes a handover note for a human agent.
    Writes: escalation_status, escalation_note, messages.
    """
    booking = state.get("booking_details") or {}
    passenger = booking.get("passenger", "the passenger")
    booking_ref = state.get("booking_ref", "N/A")
    tries = state.get("retry_count", 0)
    failure = state.get("failure_reason", "No resolution could be found automatically.")

    # Get original passenger message
    original_msg = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "N/A"
    )

    sys_msg = SystemMessage(
        content=ESCALATION_AGENT_PROMPT.format(
            passenger=passenger,
            booking_ref=booking_ref,
            original_msg=original_msg,
            tries=tries,
            failure=failure
        )
    )

    response = llm.invoke([sys_msg])
    note = response.content

    return {
        "escalation_status": True,
        "escalation_note": note,
        "messages": [response],
    }


# ── Node 7: Final Response Agent ──────────────────────────────────────────────

def final_response_agent(state: RebookingState) -> dict:
    """
    Writes the final passenger-facing reply.
    Writes: final_response, messages.
    """
    booking = state.get("booking_details") or {}
    passenger = booking.get("passenger", "there")

    if state.get("escalation_status"):
        sys_msg = SystemMessage(
            content=FINAL_RESPONSE_ESCALATED_PROMPT.format(
                passenger=passenger
            )
        )
    else:
        sol = state.get("proposed_solution", {})
        sol_type = sol.get("type", "")

        if sol_type == "rebook":
            solution_text = (
                f"a new flight has been booked: Flight {sol.get('flight_no')} "
                f"in {sol.get('cabin')} class."
            )
            next_steps = "Please check your email for the updated e-ticket."
        elif sol_type == "refund":
            solution_text = (
                f"a {sol.get('amount_description')} refund of ${sol.get('amount_usd', 0):.0f} "
                "has been approved and will be processed within 7 business days."
            )
            next_steps = "The refund will appear on your original payment method."
        elif sol_type == "compensation":
            solution_text = (
                f"compensation of ${sol.get('amount_usd', 0):.0f} has been approved. "
                f"Reason: {sol.get('reason', '')}."
            )
            next_steps = "The compensation will be credited to your account within 5 business days."
        else:
            solution_text = "your request has been processed."
            next_steps = "Please contact us if you need further assistance."

        sys_msg = SystemMessage(
            content=FINAL_RESPONSE_RESOLVED_PROMPT.format(
                passenger=passenger,
                solution_text=solution_text,
                next_steps=next_steps
            )
        )

    response = llm.invoke([sys_msg])
    content = response.content

    return {"final_response": content, "messages": [response]}
