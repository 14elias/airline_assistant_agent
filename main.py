"""
main.py — Runs all 4 test scenarios plus the Scenario 1 follow-up.

Handles two types of interrupt() pauses:
  1. clarify_constraints  — asks the passenger for travel constraints
  2. policy_checker       — asks a supervisor to approve refunds/compensation > $300

Usage:
    python main.py

Environment:
    GROQ_API_KEY must be set in .env (see .env.example).
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import build_graph

# ── Helpers ────────────────────────────────────────────────────────────────────

def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_scenario_header(scenario_id: str, message: str) -> None:
    print_separator("═")
    print(f"  SCENARIO {scenario_id}")
    print(f"  Input: \"{message}\"")
    print_separator("═")


def _print_node_update(node_name: str, updates: dict) -> None:
    """Print relevant state fields changed by a node."""
    if not updates:
        print(f"  [{node_name}]")
        return

    interesting = {}
    for key in ("intent", "constraints", "proposed_solution", "policy_check_result",
                "failure_reason", "retry_count", "escalation_status"):
        if key in updates and updates[key] not in (None, "", False, 0):
            # Only show retry_count when it's > 0
            if key == "retry_count" and updates[key] == 0:
                continue
            interesting[key] = updates[key]

    print(f"  [{node_name}]")
    for k, v in interesting.items():
        print(f"    {k}: {v}")


def _check_and_handle_interrupt(graph, config: dict) -> tuple[bool, str]:
    """
    Check if the graph is paused on an interrupt().
    If yes, print the interrupt message, get user input, and return (True, user_input).
    Returns (False, "") if the graph is not interrupted.
    """
    current_state = graph.get_state(config)

    # Check if there are pending interrupt tasks
    interrupted = False
    interrupt_value = ""
    for task in current_state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            interrupted = True
            interrupt_value = task.interrupts[0].value
            break

    if interrupted:
        print()
        print_separator("*")
        print(interrupt_value)
        print_separator("*")
        user_input = input("  Your answer: ").strip()
        return True, user_input

    return False, ""


def run_graph(graph, initial_input, config: dict) -> dict:
    """
    Stream the graph, handle any interrupt() pauses interactively, and return final state.
    `initial_input` is either a plain state dict (first call) or a Command(resume=...) (resume).
    """
    path = []

    # Stream until the graph stops (END or interrupt)
    for event in graph.stream(initial_input, config=config, stream_mode="updates"):
        for node_name, updates in event.items():
            path.append(node_name)
            _print_node_update(node_name, updates)

    print()
    print(f"  Execution path: {' → '.join(path)}")

    # ── Interrupt handling loop ────────────────────────────────────────────────
    # The graph may pause at interrupt() points. We loop until it truly ends.
    while True:
        is_interrupted, user_answer = _check_and_handle_interrupt(graph, config)
        if not is_interrupted:
            break

        # Resume with the human's answer
        resume_path = []
        for event in graph.stream(Command(resume=user_answer), config=config, stream_mode="updates"):
            for node_name, updates in event.items():
                resume_path.append(node_name)
                _print_node_update(node_name, updates)

        if resume_path:
            print()
            print(f"  (Resumed) path: {' → '.join(resume_path)}")

    final = graph.get_state(config).values
    return final


# ── Scenarios ──────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "1",
        "thread_id": "thread_sc1",
        "request_id": "R-7702",
        "booking_ref": "XK9L2P",
        "message": "My flight to Dubai was cancelled. I need to be there by tomorrow noon.",
        "expected_path": (
            "Classifier → Clarify constraints → Rebooking ⇄ Tools → Policy checker (pass) → Final response"
        ),
    },
    {
        "id": "2",
        "thread_id": "thread_sc2",
        "request_id": "R-7703",
        "booking_ref": "B77XYZ",
        "message": "My flight is delayed 6 hours. Am I entitled to anything?",
        "expected_path": "Classifier → Compensation → Policy checker (pass, $200) → Final response",
    },
    {
        "id": "3",
        "thread_id": "thread_sc3",
        "request_id": "R-7704",
        "booking_ref": "C88ABC",
        "message": "My flight was cancelled. Please rebook me.",
        "expected_path": (
            "Classifier → Clarify → Rebooking ⇄ Policy checker (fails 3×) → Escalation → Final response"
        ),
    },
    {
        "id": "4",
        "thread_id": "thread_sc4",
        "request_id": "R-7705",
        "booking_ref": "D99DEF",
        "message": "This is the third time you've cancelled on me. I want to speak to a manager.",
        "expected_path": "Classifier → Escalation → Final response",
    },
]


def run_all_scenarios(graph) -> None:
    for sc in SCENARIOS:
        print_scenario_header(sc["id"], sc["message"])
        print(f"  Expected path: {sc['expected_path']}")
        print_separator()

        config = {"configurable": {"thread_id": sc["thread_id"]}}
        initial_state = {
            "messages": [HumanMessage(content=sc["message"])],
            "request_id": sc["request_id"],
            "booking_ref": sc["booking_ref"],
            "retry_count": 0,
            "escalation_status": False,
        }

        final = run_graph(graph, initial_state, config)

        print()
        print("  ── FINAL RESPONSE ──")
        print(final.get("final_response", "(no response yet — graph may still be interrupted)"))
        print()

    # ── Scenario 1 follow-up ───────────────────────────────────────────────────
    print_scenario_header("1 FOLLOW-UP", "Actually, can I get a refund instead?")
    print("  (Continues thread_sc1 — booking_ref reused from checkpoint, no need to re-enter)")
    print_separator()

    config = {"configurable": {"thread_id": "thread_sc1"}}
    follow_up_state = {
        "messages": [HumanMessage(content="Actually, can I get a refund instead?")],
        # booking_ref, booking_details, etc. are already in the checkpointed state
    }

    final = run_graph(graph, follow_up_state, config)

    print()
    print("  ── FINAL RESPONSE ──")
    print(final.get("final_response", "(no response yet — awaiting human approval)"))
    print()


# ── Graph diagram ──────────────────────────────────────────────────────────────

def save_graph_diagram(graph) -> None:
    try:
        mermaid = graph.get_graph().draw_mermaid()
        with open("graph.mermaid", "w") as f:
            f.write(mermaid)
        print("Graph diagram saved → graph.mermaid")
    except Exception as exc:
        print(f"Could not save graph diagram: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)

    save_graph_diagram(graph)
    print()

    run_all_scenarios(graph)
