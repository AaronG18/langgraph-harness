"""A `##` heading groups the rules below it, and validation can target a group."""

from __future__ import annotations

from langgraph_harness import Harness, validate_output
from langgraph_harness.cli import EXIT_OK, EXIT_VIOLATION, main

DOC = """# Guard

- MUST: answer SAFE [require=SAFE]

## Output format
- MUST: return JSON [require=\\{]

## Tone
- MUST NOT: mention SECRET [forbid=SECRET]
- SHOULD: stay short
"""


def test_rules_carry_their_section():
    harness = Harness.from_text(DOC, name="guard")
    assert [rule.section for rule in harness.rules] == ["", "Output format", "Tone", "Tone"]


def test_h1_is_a_title_not_a_section():
    harness = Harness.from_text(DOC, name="guard")
    assert "Guard" not in harness.sections()


def test_sections_preserve_order_and_group():
    harness = Harness.from_text(DOC, name="guard")
    sections = harness.sections()
    assert list(sections) == ["", "Output format", "Tone"]
    assert [rule.id for rule in sections["Tone"]] == ["R3", "R4"]


def test_rules_in_returns_one_group():
    harness = Harness.from_text(DOC, name="guard")
    assert [rule.id for rule in harness.rules_in("Output format")] == ["R2"]
    assert harness.rules_in("Missing") == ()


def test_prompt_stays_flat_without_headings():
    flat = Harness.from_text("- MUST: a\n- SHOULD: b", name="flat")
    prompt = flat.prompt()
    assert "## " not in prompt
    assert prompt.splitlines()[2] == "- [MUST] a"


def test_prompt_names_the_sections():
    prompt = Harness.from_text(DOC, name="guard").prompt()
    assert prompt.splitlines()[0] == "HARNESS: guard"
    assert "## Output format" in prompt
    assert "## Tone" in prompt
    order = [line for line in prompt.splitlines() if line.startswith("## ") or line.startswith("- [")]
    assert order[:2] == ["- [MUST] answer SAFE", "## Output format"]


def test_validation_can_be_scoped_to_a_section():
    harness = Harness.from_text(DOC, name="guard")

    # Only the Tone rules: a secret fails, the missing JSON does not.
    tone = validate_output("SECRET no json", harness, sections=["Tone"])
    assert {v.rule_id for v in tone.violations} == {"R3"}

    # Only the Output format rule: no brace, violation.
    fmt = validate_output("SAFE text", harness, sections=["Output format"])
    assert {v.rule_id for v in fmt.violations} == {"R2"}

    # No filter keeps the previous behaviour.
    everything = validate_output("SAFE {}", harness)
    assert everything.violations == ()


def test_violation_records_the_section():
    harness = Harness.from_text(DOC, name="guard")
    report = validate_output("SAFE {} SECRET", harness, sections=["Tone"])
    assert report.violations[0].section == "Tone"
    assert report.as_dicts()[0]["section"] == "Tone"


def test_cli_section_option(tmp_path, capsys):
    harness = tmp_path / "guard.md"
    harness.write_text(DOC, encoding="utf-8")

    # Fails the Output format rule but not Tone.
    assert main(["check", str(harness), "--text", "SAFE {}", "--section", "Tone"]) == EXIT_OK
    assert "OK" in capsys.readouterr().out

    # Same text, restricted to Tone, still passes; a secret now fails.
    assert main(["check", str(harness), "--text", "SAFE {} SECRET", "--section", "Tone"]) == EXIT_VIOLATION
    assert "VIOLATION R3" in capsys.readouterr().out
