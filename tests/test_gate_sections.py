"""A gate can listen to one section only, and warnings never open it."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from langgraph_harness import (
    Harness,
    HarnessState,
    harness_gate,
    harnessed,
    violations_of,
)

DOC = """# guard

## Output format
- MUST: return JSON [require=\\{]

## Tone
- MUST NOT: mention SECRET [forbid=SECRET]
"""


def _recorded(*sections_and_ids):
    return {"harness_violations": [{"rule_id": rid, "section": sec} for sec, rid in sections_and_ids]}


def test_violations_of_filters_by_section():
    state = _recorded(("Output format", "R1"), ("Tone", "R2"))
    assert [v["rule_id"] for v in violations_of(state, sections=["Tone"])] == ["R2"]
    assert [v["rule_id"] for v in violations_of(state, sections=[])] == []
    assert len(violations_of(state)) == 2


def test_violations_of_treats_missing_section_as_preamble():
    state = {"harness_violations": [{"rule_id": "old"}]}
    assert [v["rule_id"] for v in violations_of(state, sections=[""])] == ["old"]
    assert violations_of(state, sections=["Tone"]) == []


def test_gate_routes_only_for_the_listed_section():
    route = harness_gate(on_violation="repair", sections=["Tone"])

    assert route(_recorded(("Tone", "R2"))) == "repair"
    assert route(_recorded(("Output format", "R1"))) == END
    assert route(_recorded(("Output format", "R1"), ("Tone", "R2"))) == "repair"


def test_gate_without_sections_keeps_old_behaviour():
    route = harness_gate(on_violation="repair")
    assert route(_recorded(("Output format", "R1"))) == "repair"


def test_gate_accepts_the_preamble_section():
    route = harness_gate(on_violation="repair", sections=[""])
    assert route({"harness_violations": [{"rule_id": "R0"}]}) == "repair"
    assert route(_recorded(("Tone", "R2"))) == END


def test_gate_sections_combine_with_max_retries():
    route = harness_gate(on_violation="repair", max_retries=1, on_exhausted="give_up", sections=["Tone"])
    full = {**_recorded(("Tone", "R2")), "harness_repair_count": 1}
    assert route(full) == "give_up"
    # An out-of-scope violation must not be pushed into the retry branch either.
    assert route({**_recorded(("Output format", "R1")), "harness_repair_count": 1}) == END


class State(HarnessState, total=False):
    messages: list
    repaired: bool


def build(node_body: str, gate_sections):
    harness = Harness.from_text(DOC, name="guard")

    @harnessed(harness, validate=True, on_violation="record", count_repairs=True)
    def node(state):
        return {"messages": [node_body]}

    def repair(state):
        return {"repaired": True, "messages": ["{}"]}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_node("repair", repair)
    builder.add_edge(START, "node")
    builder.add_conditional_edges(
        "node",
        harness_gate(on_violation="repair", sections=gate_sections),
    )
    builder.add_edge("repair", END)
    return builder.compile()


def test_graph_ignores_a_violation_outside_the_gated_section():
    # Fails "Output format" only, but the gate listens to "Tone": no repair.
    out = build("plain text", ["Tone"]).invoke({"messages": []})
    assert out.get("repaired") is None
    assert out["harness_violations"][0]["section"] == "Output format"


def test_graph_repairs_when_the_gated_section_fails():
    # Fails "Tone" (mentions SECRET), gate listens to "Tone": repair runs.
    out = build("{} SECRET", ["Tone"]).invoke({"messages": []})
    assert out["repaired"] is True


def test_should_warning_never_opens_the_gate():
    harness = Harness.from_text("- SHOULD: mention SAFE [require=SAFE]", name="soft")

    @harnessed(harness, validate=True, on_violation="record", count_repairs=True)
    def soft(state):
        return {"messages": ["nothing"]}

    class S(HarnessState, total=False):
        messages: list
        repaired: bool

    def repair(state):
        return {"repaired": True, "messages": ["SAFE"]}

    builder = StateGraph(S)
    builder.add_node("soft", soft)
    builder.add_node("repair", repair)
    builder.add_edge(START, "soft")
    builder.add_conditional_edges("soft", harness_gate(on_violation="repair"))
    builder.add_edge("repair", END)

    out = builder.compile().invoke({"messages": []})
    assert out.get("repaired") is None
    assert violations_of(out) == []
