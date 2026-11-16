from langgraph.graph import END

from langgraph_harness import HarnessState, harness_gate, violations_of, with_harness


def test_violations_of_dict_and_empty():
    assert violations_of({"harness_violations": [{"rule_id": "R1"}]})[0]["rule_id"] == "R1"
    assert violations_of({}) == []


def test_with_harness_copies_state():
    original = {"harness": {}}
    out = with_harness(original, "n", "text")

    assert out["harness"]["n"] == "text"
    assert original["harness"] == {}


def test_gate_routes_on_violations():
    route = harness_gate(on_violation="repair")

    assert route({"harness_violations": [{"rule_id": "R1"}]}) == "repair"
    assert route({}) == END


def test_state_typeddict_has_channels():
    assert "harness" in HarnessState.__annotations__
    assert "harness_violations" in HarnessState.__annotations__
