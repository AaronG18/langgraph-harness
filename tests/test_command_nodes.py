"""A node may return a Command; validation must look inside its update, not skip it."""

from __future__ import annotations

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langgraph_harness import (
    Harness,
    HarnessState,
    HarnessViolation,
    harnessed,
    violations_of,
)

GUARD = Harness.from_text("- MUST: answer SAFE [require=SAFE]", name="guard")


class State(HarnessState, total=False):
    messages: list
    repaired: bool


def test_command_is_validated_and_can_raise():
    @harnessed(GUARD, validate=True)
    def node(state):
        return Command(update={"messages": ["BAD"]}, goto="next")

    with pytest.raises(HarnessViolation):
        node({"messages": []})


def test_command_records_violations_in_its_own_update():
    @harnessed(GUARD, validate=True, on_violation="record")
    def node(state):
        return Command(update={"messages": ["BAD"]}, goto="next")

    out = node({"messages": []})

    assert isinstance(out, Command)
    assert out.goto == "next"
    assert out.update["messages"] == ["BAD"]
    assert out.update["harness_violations"][0]["rule_id"] == "R1"


def test_command_preserves_graph_and_goto():
    @harnessed(GUARD, validate=True, on_violation="record")
    def node(state):
        return Command(graph="parent", update={"messages": ["BAD"]}, goto=("a", "b"))

    out = node({"messages": []})

    assert out.graph == "parent"
    assert out.goto == ("a", "b")


def test_command_increments_the_repair_counter():
    @harnessed(GUARD, validate=True, on_violation="record", count_repairs=True)
    def node(state):
        return Command(update={"messages": ["BAD"]})

    first = node({"messages": []})
    second = node({"messages": [], "harness_repair_count": 4})

    assert first.update["harness_repair_count"] == 1
    assert second.update["harness_repair_count"] == 5


def test_compliant_command_clears_stale_violations():
    @harnessed(GUARD, validate=True, on_violation="record")
    def node(state):
        return Command(update={"messages": ["SAFE"]}, goto="next")

    out = node({"messages": [], "harness_violations": [{"rule_id": "R1"}]})

    assert out.update["harness_violations"] == []
    assert out.update["messages"] == ["SAFE"]
    assert out.goto == "next"


def test_command_without_update_is_skipped():
    @harnessed(GUARD, validate=True)
    def node(state):
        return Command(goto="next")

    out = node({"messages": []})
    assert isinstance(out, Command)
    assert out.update is None


def test_graph_routes_through_command_with_violations_recorded():
    @harnessed(GUARD, validate=True, on_violation="record", count_repairs=True)
    def node(state):
        return Command(update={"messages": ["BAD"]}, goto="repair")

    def repair(state):
        return {"repaired": True, "messages": ["SAFE"]}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_node("repair", repair)
    builder.add_edge(START, "node")
    builder.add_edge("repair", END)

    out = builder.compile().invoke({"messages": []})

    assert out["repaired"] is True
    assert out["messages"] == ["SAFE"]
    assert violations_of(out)[0]["rule_id"] == "R1"
    assert out["harness_repair_count"] == 1
