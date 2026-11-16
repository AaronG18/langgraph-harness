"""End-to-end test at the LLM-call boundary, without a real provider.

Each node builds the model input from ``harness_prompt()``; the fake model only
knows the harness it was handed. If a node received the wrong document (or no
document), the assertions below would fail.
"""

from langgraph.graph import END, START, StateGraph

from langgraph_harness import (
    Harness,
    HarnessState,
    current_harness,
    harness_prompt,
    harnessed,
)

PLANNER = Harness.from_text("- MUST: answer PLANNER [require=PLANNER]", name="planner")
WRITER = Harness.from_text("- MUST: answer WRITER [require=WRITER]", name="writer")


class State(HarnessState):
    messages: list
    outputs: dict


def fake_model(system: str, user: str) -> str:
    del user
    if "HARNESS: planner" in system:
        return "PLANNER"
    if "HARNESS: writer" in system:
        return "WRITER"
    return "UNKNOWN"


def test_each_node_calls_the_model_with_its_own_harness():
    @harnessed(PLANNER, validate=True)
    def planner(state: State):
        answer = fake_model(harness_prompt(), "?")
        return {
            "messages": [answer],
            "outputs": {**state.get("outputs", {}), "planner": answer, "planner_harness": current_harness().name},
        }

    @harnessed(WRITER, validate=True)
    def writer(state: State):
        answer = fake_model(harness_prompt(), "?")
        return {
            "messages": [answer],
            "outputs": {**state.get("outputs", {}), "writer": answer, "writer_harness": current_harness().name},
        }

    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("writer", writer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "writer")
    builder.add_edge("writer", END)

    out = builder.compile().invoke({"messages": [], "outputs": {}})

    assert out["outputs"]["planner"] == "PLANNER"
    assert out["outputs"]["writer"] == "WRITER"
    assert out["outputs"]["planner_harness"] == "planner"
    assert out["outputs"]["writer_harness"] == "writer"
