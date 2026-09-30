"""
main.py — Runs all 4 test scenarios plus the Scenario 1 follow-up.

Usage:
    python main.py

Environment:
    XAI_API_KEY must be set in .env (see .env.example).
"""
import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph

# ── Helpers ────────────────────────────────────────────────────────────────────

def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_scenario_header(scenario_id: str, message: str) -> None:
    print_separator("═")
    print(f"  SCENARIO {scenario_id}")
    print(f"  Input: \"{message}\"")
    print_separator("═")


def run_graph(graph, initial_state: dict, config: dict) -> dict:
    """Stream the graph and print the execution path. Returns the final state."""
    path = []
    for event in graph.stream(initial_state, config=config):
        for node_name, updates in event.items():
            path.append(node_name)
            # Show key state changes per node
            interesting = {}
            if "intent" in updates:
                interesting["intent"] = updates["intent"]
            if "constraints" in updates:
                interesting["constraints"] = updates["constraints"]
            if "proposed_solution" in updates and updates["proposed_solution"]:
                interesting["proposed_solution"] = updates["proposed_solution"]
            if "policy_check_result" in updates:
                interesting["policy_check_result"] = updates["policy_check_result"]
            if "failure_reason" in updates and updates["failure_reason"]:
                interesting["failure_reason"] = updates["failure_reason"]
            if "retry_count" in updates:
                interesting["retry_count"] = updates["retry_count"]
            if "escalation_status" in updates and updates["escalation_status"]:
                interesting["escalation_status"] = updates["escalation_status"]

            if interesting:
                print(f"  [{node_name}]")
                for k, v in interesting.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  [{node_name}]")

    print()
    print(f"  Execution path: {' → '.join(path)}")

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
        "expected_path": "Classifier → Rebooking ⇄ Tools → Policy checker (pass) → Final response",
    },
    {
        "id": "2",
        "thread_id": "thread_sc2",
        "request_id": "R-7703",
        "booking_ref": "B77XYZ",
        "message": "My flight is delayed 6 hours. Am I entitled to anything?",
        "expected_path": "Classifier → Compensation → Policy checker (pass) → Final response",
    },
    {
        "id": "3",
        "thread_id": "thread_sc3",
        "request_id": "R-7704",
        "booking_ref": "C88ABC",
        "message": "My flight was cancelled. I need to be rebooked.",
        "expected_path": "Classifier → Rebooking ⇄ Policy checker (fails 3×) → Escalation → Final response",
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


def run_all_scenarios(graph, checkpointer: MemorySaver) -> None:
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
        print(final.get("final_response", "(no response)"))
        print()

    # ── Scenario 1 follow-up ───────────────────────────────────────────────
    print_scenario_header("1 FOLLOW-UP", "Actually, can I get a refund instead?")
    print("  (Continues thread_sc1 — no booking_ref needed; state is persisted)")
    print_separator()

    config = {"configurable": {"thread_id": "thread_sc1"}}
    follow_up_state = {
        "messages": [HumanMessage(content="Actually, can I get a refund instead?")],
        # NOTE: booking_ref, intent, booking_details already in checkpointed state
    }

    final = run_graph(graph, follow_up_state, config)

    print()
    print("  ── FINAL RESPONSE ──")
    print(final.get("final_response", "(no response)"))
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

    run_all_scenarios(graph, checkpointer)
