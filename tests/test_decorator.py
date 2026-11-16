import asyncio

import pytest

from langgraph_harness import (
    HarnessNotBound,
    HarnessViolation,
    current_harness,
    harnessed,
    parse_harness,
)

HARNESS = parse_harness("- MUST: ok [require=ok]", name="h")


def test_current_harness_bound_inside_and_reset_after():
    with pytest.raises(HarnessNotBound):
        current_harness()

    @harnessed(HARNESS)
    def node(state):
        return current_harness().name

    assert node({}) == "h"

    with pytest.raises(HarnessNotBound):
        current_harness()


def test_validate_raises_by_default():
    @harnessed(HARNESS, validate=True)
    def node(state):
        return "bad"

    with pytest.raises(HarnessViolation) as exc:
        node({})

    assert exc.value.harness_name == "h"


def test_validate_records_in_dict_result():
    @harnessed(HARNESS, validate=True, on_violation="record")
    def node(state):
        return {"messages": ["bad"]}

    out = node({})
    assert out["harness_violations"][0]["rule_id"] == "R1"


def test_validate_passes():
    @harnessed(HARNESS, validate=True)
    def node(state):
        return "ok"

    assert node({}) == "ok"


def test_async_node_is_supported():
    @harnessed(HARNESS)
    async def node(state):
        return current_harness().name

    assert asyncio.run(node({})) == "h"


def test_wrapper_exposes_metadata():
    @harnessed(HARNESS, name="planner")
    def node(state):
        return None

    assert node.__harness_name__ == "h"
    assert node.__harness_node__ == "planner"
    assert node.__harness__.name == "h"


def test_inject_messages_prepends_system_message():
    @harnessed(HARNESS, inject_messages=True)
    def node(state):
        return state["messages"][0].content

    text = node({"messages": ["hi"]})
    assert "HARNESS: h" in text


def test_invalid_on_violation():
    with pytest.raises(ValueError):
        harnessed(HARNESS, on_violation="explode")
