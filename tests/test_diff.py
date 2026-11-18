"""Harness.diff answers "what changed in my rules file?" between two revisions."""

from __future__ import annotations

import json

from langgraph_harness import Harness
from langgraph_harness.cli import EXIT_ERROR, EXIT_OK, EXIT_VIOLATION, main

BASE = """# guard

## Output format
- MUST: return JSON [require=\\{]

## Tone
- MUST NOT: mention SECRET [forbid=SECRET]
"""


def test_identical_harnesses_have_no_changes():
    old = Harness.from_text(BASE, name="a")
    new = Harness.from_text(BASE, name="b")

    result = old.diff(new)
    assert result.is_empty is True
    assert len(result) == 0
    assert result.format() == "no changes"


def test_added_rule():
    new_text = BASE + "- SHOULD: keep it short\n"
    result = Harness.from_text(BASE).diff(Harness.from_text(new_text))

    assert [r.text for r in result.added] == ["keep it short"]
    assert result.removed == ()
    assert result.changed == ()
    assert "+ [SHOULD] (Tone) keep it short" in result.format()


def test_removed_rule():
    new_text = "# guard\n\n## Output format\n- MUST: return JSON [require=\\{]\n"
    result = Harness.from_text(BASE).diff(Harness.from_text(new_text))

    assert [r.text for r in result.removed] == ["mention SECRET"]
    assert result.added == ()


def test_level_change_is_reported_as_changed():
    new_text = BASE.replace("- MUST NOT: mention SECRET", "- SHOULD: mention SECRET")
    result = Harness.from_text(BASE).diff(Harness.from_text(new_text))

    assert len(result.changed) == 1
    old_rule, new_rule = result.changed[0]
    assert old_rule.level == "MUST NOT"
    assert new_rule.level == "SHOULD"
    assert "MUST NOT -> SHOULD" in result.format()


def test_annotation_change_is_reported_as_changed():
    new_text = BASE.replace("[require=\\{]", "[require=steps]")
    result = Harness.from_text(BASE).diff(Harness.from_text(new_text))

    assert len(result.changed) == 1
    assert "require changed" in result.format()


def test_moving_a_rule_between_sections_is_remove_plus_add():
    new_text = BASE.replace("## Tone\n- MUST NOT: mention SECRET [forbid=SECRET]\n",
                            "- MUST NOT: mention SECRET [forbid=SECRET]\n\n## Tone\n- SHOULD: keep it short\n")
    result = Harness.from_text(BASE).diff(Harness.from_text(new_text))

    assert [r.section for r in result.removed] == ["Tone"]
    # The rule survives under a different heading, so it is a removal plus an addition.
    assert sorted(r.section for r in result.added) == ["Output format", "Tone"]


def test_as_dict_shape():
    new_text = BASE + "- SHOULD: extra\n"
    payload = Harness.from_text(BASE).diff(Harness.from_text(new_text)).as_dict()

    assert payload["added"] == [{"section": "Tone", "level": "SHOULD", "text": "extra"}]
    assert payload["removed"] == []
    assert payload["changed"] == []


def test_cli_diff_exit_codes_and_output(tmp_path, capsys):
    old = tmp_path / "old.md"
    same = tmp_path / "same.md"
    newer = tmp_path / "new.md"
    old.write_text(BASE, encoding="utf-8")
    same.write_text(BASE, encoding="utf-8")
    newer.write_text(BASE + "- SHOULD: extra\n", encoding="utf-8")

    assert main(["diff", str(old), str(same)]) == EXIT_OK
    assert "no changes" in capsys.readouterr().out

    assert main(["diff", str(old), str(newer)]) == EXIT_VIOLATION
    assert "+ [SHOULD] (Tone) extra" in capsys.readouterr().out

    assert main(["diff", str(tmp_path / "missing.md"), str(newer)]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_cli_diff_json(tmp_path, capsys):
    old = tmp_path / "old.md"
    newer = tmp_path / "new.md"
    old.write_text(BASE, encoding="utf-8")
    newer.write_text(BASE + "- SHOULD: extra\n", encoding="utf-8")

    assert main(["diff", str(old), str(newer), "--json"]) == EXIT_VIOLATION
    payload = json.loads(capsys.readouterr().out)
    assert payload["different"] is True
    assert payload["count"] == 1
    assert payload["added"][0]["text"] == "extra"
