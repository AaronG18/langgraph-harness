import pytest

from langgraph_harness import parse_harness, validate_output

HARNESS = parse_harness(
    '- MUST: json [require="steps"]\n'
    "- MUST NOT: tool [forbid=tool_]\n"
    "- SHOULD: short [require=short]",
    name="h",
)


def test_pass_and_should_is_only_a_warning():
    report = validate_output('{"steps": []}', HARNESS)

    assert report.ok
    assert report.violations == ()
    assert any(warning.level == "SHOULD" for warning in report.warnings)


def test_must_require_failure():
    report = validate_output("nothing", HARNESS)

    assert not report.ok
    assert any(v.rule_id == "R1" for v in report.violations)


def test_must_not_forbid_failure():
    report = validate_output('{"steps": []} tool_call', HARNESS)

    assert not report.ok
    assert any(v.rule_id == "R2" for v in report.violations)


def test_non_string_raises():
    with pytest.raises(TypeError):
        validate_output(123, HARNESS)  # type: ignore[arg-type]


def test_report_as_dicts():
    report = validate_output("nothing", HARNESS)
    assert report.as_dicts()[0]["rule_id"] == "R1"
