"""
graph.py — Builds the LangGraph StateGraph for the Flight Disruption Rebooking Assistant.

Graph topology (9 nodes, 4 conditional edges, 1 cycle, 2 interrupt() points):

    START
      │
      ▼
  request_classifier ──(route_intent)──► clarify_constraints ──(route_after_clarify)──► rebooking_agent ◄──┐
                      ├──────────────────────────────────────────────────────────────► refund_agent       ◄──┤
                      ├──────────────────────────────────────────────────────────────► compensation_agent ◄──┤
                      └──────────────────────────────────────────────────────────────► escalation_agent      │
                                                                                                              │
  clarify_constraints uses interrupt() if constraints are vague (rebook only).                                │
                                                                                                              │
      ┌────────────────────────────────────────────────────────────────────────────────────────────────────  ┘
      │
  [tools_condition]
      ├── tool_calls present → tools (ToolNode) ──► back to same agent (route_back_from_tools)
      └── no tool_calls      → policy_checker
                                    │
                          (route_policy)
                              ├── pass → final_response_agent → END
                              │          (interrupt() if amount > $300 — human-in-the-loop)
                              ├── fail, retry_count < 3 → back to same agent (cycle)
                              └── fail, retry_count >= 3 → escalation_agent → final_response_agent → END

interrupt() points:
  1. clarify_constraints  — asks passenger for travel constraints when vague
  2. policy_checker       — asks supervisor to approve refund/compensation > $300
"""
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

from state import RebookingState
from tools import get_booking, search_flights, get_fare_rules
from nodes import (
    request_classifier,
    clarify_constraints,
    rebooking_agent,
    refund_agent,
    compensation_agent,
    policy_checker,
    escalation_agent,
    final_response_agent,
)

# ── Routing functions ──────────────────────────────────────────────────────────

def route_intent(state: RebookingState) -> str:
    """
    Conditional edge #1 — after request_classifier.
    Rebook intent goes to clarify_constraints first (bonus feature).
    Other intents go directly to their specialist.
    """
    intent = state.get("intent", "complaint").lower().strip()
    if intent == "rebook":
        return "clarify_constraints"     # ← bonus: check constraints first
    elif intent == "refund":
        return "refund_agent"
    elif intent == "compensation":
        return "compensation_agent"
    else:
        return "escalation_agent"


def route_after_clarify(state: RebookingState) -> str:
    """
    Conditional edge #2 — after clarify_constraints.
    Always routes to rebooking_agent (clarification is done).
    """
    return "rebooking_agent"


def route_back_from_tools(state: RebookingState) -> str:
    """
    Conditional edge #3 — after ToolNode.
    Routes back to whichever specialist agent called the tool.
    """
    return state.get("active_agent", "rebooking_agent")


def route_policy(state: RebookingState) -> str:
    """
    Conditional edge #4 — after policy_checker.
    - pass → final_response_agent
    - fail + retry_count < 3 → retry with same specialist agent (the cycle)
    - fail + retry_count >= 3 → escalation_agent (guarantees graph terminates)
    """
    if state.get("policy_check_result") == "pass":
        return "final_response_agent"
    elif state.get("retry_count", 0) >= 3:
        return "escalation_agent"
    else:
        return state.get("active_agent", "rebooking_agent")


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and returns the uncompiled StateGraph builder.
    Call .compile(checkpointer=...) on the returned builder to get a runnable graph.
    """
    builder = StateGraph(RebookingState)

    # ── Register nodes ────────────────────────────────────────────────
    builder.add_node("request_classifier", request_classifier)
    builder.add_node("clarify_constraints", clarify_constraints)   # bonus node
    builder.add_node("rebooking_agent", rebooking_agent)
    builder.add_node("refund_agent", refund_agent)
    builder.add_node("compensation_agent", compensation_agent)
    builder.add_node("policy_checker", policy_checker)
    builder.add_node("escalation_agent", escalation_agent)
    builder.add_node("final_response_agent", final_response_agent)

    # ToolNode handles all tool calls for all three specialist agents
    all_tools = [get_booking, search_flights, get_fare_rules]
    builder.add_node("tools", ToolNode(all_tools))

    # ── Edges ─────────────────────────────────────────────────────────

    # Entry point
    builder.add_edge(START, "request_classifier")

    # Conditional edge #1: classifier → clarify (rebook) or specialist
    builder.add_conditional_edges(
        "request_classifier",
        route_intent,
        {
            "clarify_constraints": "clarify_constraints",
            "refund_agent": "refund_agent",
            "compensation_agent": "compensation_agent",
            "escalation_agent": "escalation_agent",
        },
    )

    # Conditional edge #2: clarify → rebooking_agent (always)
    builder.add_conditional_edges(
        "clarify_constraints",
        route_after_clarify,
        {"rebooking_agent": "rebooking_agent"},
    )

    # Specialist agents → tools (tool_calls present) or → policy_checker
    # Uses the prebuilt tools_condition as required by the spec
    for agent_node in ("rebooking_agent", "refund_agent", "compensation_agent"):
        builder.add_conditional_edges(
            agent_node,
            tools_condition,                    # ← prebuilt tools_condition
            {"tools": "tools", END: "policy_checker"},
        )

    # Conditional edge #3: ToolNode → back to same specialist
    builder.add_conditional_edges(
        "tools",
        route_back_from_tools,
        {
            "rebooking_agent": "rebooking_agent",
            "refund_agent": "refund_agent",
            "compensation_agent": "compensation_agent",
        },
    )

    # Conditional edge #4: policy_checker → pass/fail/escalate (contains the cycle)
    builder.add_conditional_edges(
        "policy_checker",
        route_policy,
        {
            "final_response_agent": "final_response_agent",
            "rebooking_agent": "rebooking_agent",
            "refund_agent": "refund_agent",
            "compensation_agent": "compensation_agent",
            "escalation_agent": "escalation_agent",
        },
    )

    # Linear edges
    builder.add_edge("escalation_agent", "final_response_agent")
    builder.add_edge("final_response_agent", END)

    return builder
