"""A repair loop must stop at ``max_retries`` instead of hitting recursion_limit."""

import pytest
from langgraph.graph import END, START, StateGraph

from langgraph_harness import (
    Harness,
    HarnessState,
    harness_gate,
    harnessed,
    record_repair_attempt,
    repair_count_of,
)

GUARD = Harness.from_text("- MUST: answer SAFE [require=SAFE]", name="guard")


def test_gate_without_violations_goes_otherwise():
    route = harness_gate(on_violation="repair", max_retries=2)
    assert route({"harness_violations": []}) == END


def test_gate_without_max_retries_keeps_old_behaviour():
    route = harness_gate(on_violation="repair")
    assert route({"harness_violations": [{"rule_id": "R1"}], "harness_repair_count": 99}) == "repair"


def test_gate_stops_at_max_retries():
    route = harness_gate(on_violation="repair", max_retries=2, on_exhausted="give_up")

    assert route({"harness_violations": [{"rule_id": "R1"}], "harness_repair_count": 0}) == "repair"
    assert route({"harness_violations": [{"rule_id": "R1"}], "harness_repair_count": 1}) == "repair"
    assert route({"harness_violations": [{"rule_id": "R1"}], "harness_repair_count": 2}) == "give_up"


def test_gate_max_retries_zero_goes_straight_to_exhausted():
    route = harness_gate(on_violation="repair", max_retries=0, on_exhausted="give_up")
    assert route({"harness_violations": [{"rule_id": "R1"}], "harness_repair_count": 0}) == "give_up"


def test_gate_rejects_negative_max_retries():
    with pytest.raises(ValueError):
        harness_gate(on_violation="repair", max_retries=-1)


def test_count_repairs_increments_from_input_state():
    @harnessed(GUARD, validate=True, on_violation="record", count_repairs=True)
    def node(state):
        return {"messages": ["BAD"]}

    first = node({"messages": []})
    assert first["harness_repair_count"] == 1

    second = node({"messages": [], "harness_repair_count": 1})
    assert second["harness_repair_count"] == 2


def test_count_repairs_is_off_by_default():
    @harnessed(GUARD, validate=True, on_violation="record")
    def node(state):
        return {"messages": ["BAD"]}

    assert "harness_repair_count" not in node({"messages": []})


def test_record_repair_attempt_helper():
    assert record_repair_attempt({}) == {"harness_repair_count": 1}
    assert record_repair_attempt({"harness_repair_count": 4}) == {"harness_repair_count": 5}
    assert repair_count_of({"harness_repair_count": "3"}) == 3
    assert repair_count_of({}) == 0


class State(HarnessState, total=False):
    messages: list
    fixed: bool
    exhausted: bool


@harnessed(GUARD, validate=True, on_violation="record", count_repairs=True)
def always_bad(state):
    return {"messages": ["BAD"]}


def repair_that_fails(state):
    return {"messages": ["STILL BAD"]}


def give_up(state):
    return {"exhausted": True}


def build_failing_loop_graph():
    builder = StateGraph(State)
    builder.add_node("bad", always_bad)
    builder.add_node("repair", repair_that_fails)
    builder.add_node("give_up", give_up)
    builder.add_edge(START, "bad")
    builder.add_conditional_edges(
        "bad",
        harness_gate(on_violation="repair", max_retries=2, on_exhausted="give_up"),
    )
    builder.add_edge("repair", "bad")
    builder.add_edge("give_up", END)
    return builder.compile()


def test_repair_loop_terminates_instead_of_recursion_error():
    out = build_failing_loop_graph().invoke({"messages": []})

    assert out["exhausted"] is True
    assert out["harness_repair_count"] == 2
    # harness_violations holds the violations of the latest attempt, not a growing log.
    assert len(out["harness_violations"]) >= 1


@harnessed(GUARD, validate=True, on_violation="record", count_repairs=True)
def bad_until_fixed(state):
    if state.get("fixed"):
        return {"messages": ["SAFE"]}
    return {"messages": ["BAD"]}


def fix_it(state):
    return {"fixed": True, "messages": ["SAFE"]}


def test_successful_repair_ends_without_exhausting():
    builder = StateGraph(State)
    builder.add_node("bad", bad_until_fixed)
    builder.add_node("repair", fix_it)
    builder.add_node("give_up", give_up)
    builder.add_edge(START, "bad")
    builder.add_conditional_edges(
        "bad",
        harness_gate(on_violation="repair", max_retries=2, on_exhausted="give_up"),
    )
    builder.add_edge("repair", "bad")
    builder.add_edge("give_up", END)
    graph = builder.compile()

    out = graph.invoke({"messages": []})

    assert out.get("exhausted") is None
    assert out["fixed"] is True
    assert out["harness_repair_count"] == 1
