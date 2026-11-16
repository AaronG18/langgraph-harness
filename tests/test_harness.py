import pytest

from langgraph_harness import Harness, load_harness, parse_harness
from langgraph_harness.errors import HarnessParseError

SAMPLE = """
# Planner harness

- MUST: return a JSON object [require="steps"]
- MUST NOT: mention internal tool names [forbid=tool_]
- SHOULD: keep the plan short
"""


def test_parse_levels_and_annotations():
    harness = parse_harness(SAMPLE, name="planner")

    assert harness.name == "planner"
    assert len(harness) == 3

    must = harness.rules_for("MUST")
    assert len(must) == 1
    assert must[0].require == '"steps"'
    assert must[0].forbid is None
    assert must[0].is_hard is True

    must_not = harness.rules_for("MUST NOT")
    assert must_not[0].forbid == "tool_"

    should = harness.rules_for("SHOULD")
    assert should[0].is_hard is False

    prompt = harness.prompt()
    assert "HARNESS: planner" in prompt
    assert "- [MUST]" in prompt


def test_parse_ignores_non_rule_lines():
    harness = parse_harness("intro\n## Rules\n- MUST: a\n- not a rule\n", name="x")
    assert len(harness) == 1
    assert harness.rules[0].text == "a"


def test_rules_for_rejects_unknown_level():
    harness = parse_harness("- MUST: a", name="x")
    with pytest.raises(ValueError):
        harness.rules_for("MAY")


def test_load_from_file_path_and_object(tmp_path):
    path = tmp_path / "planner.md"
    path.write_text("- MUST: do it", encoding="utf-8")

    assert load_harness(str(path)).name == "planner"
    assert load_harness(path).name == "planner"

    obj = Harness.from_text("- MUST: x", name="x")
    assert load_harness(obj) is obj


def test_parse_non_string_raises():
    with pytest.raises(HarnessParseError):
        parse_harness(123)  # type: ignore[arg-type]


def test_from_text_default_name():
    assert Harness.from_text("- MUST: a").name == "harness"


def test_empty_harness_prompt():
    harness = Harness.from_text("# nothing here", name="empty")
    assert len(harness) == 0
    assert "no machine-readable rules" in harness.prompt()
