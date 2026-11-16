"""Two LangGraph nodes, each parsing its own HARNESS.

Run: python examples/two_nodes.py
"""

from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langgraph_harness import Harness, HarnessState, current_harness, harnessed

PLANNER = Harness.from_text(
    """
# Planner
- MUST: produce a numbered plan
- SHOULD: keep it under five steps
""",
    name="planner",
)

WRITER = Harness.from_text(
    """
# Writer
- MUST: write in the voice of the plan
- MUST NOT: invent new steps
""",
    name="writer",
)


class State(HarnessState):
    messages: Annotated[list, add_messages]
    trace: list[str]


@harnessed(PLANNER)
def planner(state):
    harness = current_harness()
    print(f"[planner] parsed its own harness: {harness.name} ({len(harness)} rules)")
    print(harness.prompt())
    return {"trace": [*state.get("trace", []), harness.name]}


@harnessed(WRITER)
def writer(state):
    harness = current_harness()
    print(f"[writer] parsed its own harness: {harness.name} ({len(harness)} rules)")
    print(harness.prompt())
    return {"trace": [*state.get("trace", []), harness.name]}


def main() -> None:
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("writer", writer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "writer")
    builder.add_edge("writer", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [], "trace": []})
    print("trace:", out["trace"])
    assert out["trace"] == ["planner", "writer"]


if __name__ == "__main__":
    main()
