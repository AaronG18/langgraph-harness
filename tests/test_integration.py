from langgraph.graph import END, START, StateGraph

from langgraph_harness import (
    Harness,
    HarnessState,
    current_harness,
    harness_gate,
    harnessed,
)

PLANNER = Harness.from_text("- MUST: plan", name="planner")
WRITER = Harness.from_text("- MUST: write", name="writer")


class State(HarnessState):
    seen: list[str]


@harnessed(PLANNER)
def planner_node(state):
    return {"seen": [*state.get("seen", []), current_harness().name]}


@harnessed(WRITER)
def writer_node(state):
    return {"seen": [*state.get("seen", []), current_harness().name]}


def test_each_node_sees_its_own_harness():
    builder = StateGraph(State)
    builder.add_node("planner", planner_node)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "writer")
    builder.add_edge("writer", END)
    graph = builder.compile()

    out = graph.invoke({"seen": []})

    assert out["seen"] == ["planner", "writer"]


STRICT = Harness.from_text('- MUST: json [require="ok"]', name="strict")


class GateState(HarnessState, total=False):
    messages: list
    repaired: bool


@harnessed(STRICT, validate=True, on_violation="record")
def strict_node(state):
    return {"messages": ["nope"]}


def repair_node(state):
    return {"repaired": True}


def test_edge_gate_routes_to_repair():
    builder = StateGraph(GateState)
    builder.add_node("strict", strict_node)
    builder.add_node("repair", repair_node)
    builder.add_edge(START, "strict")
    builder.add_conditional_edges("strict", harness_gate(on_violation="repair", otherwise=END))
    builder.add_edge("repair", END)
    graph = builder.compile()

    out = graph.invoke({"messages": []})

    assert out["repaired"] is True
    assert out["harness_violations"][0]["rule_id"] == "R1"
