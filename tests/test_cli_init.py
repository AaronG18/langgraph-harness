"""`init` must produce a document that parses, groups, and validates out of the box."""

from __future__ import annotations

from langgraph_harness import Harness
from langgraph_harness.cli import EXIT_ERROR, EXIT_OK, EXIT_VIOLATION, main


def test_init_writes_a_parseable_starter(tmp_path, capsys):
    target = tmp_path / "planner.md"

    assert main(["init", str(target)]) == EXIT_OK
    assert target.is_file()
    assert "wrote" in capsys.readouterr().out

    harness = Harness.from_file(target)
    assert harness.name == "planner"
    assert [rule.section for rule in harness.rules] == ["Output format", "Tone", "Tone"]


def test_starter_round_trips_through_check(tmp_path):
    target = tmp_path / "guard.md"
    assert main(["init", str(target)]) == EXIT_OK

    assert main(["check", str(target), "--text", '{"steps": []}']) == EXIT_OK
    assert main(["check", str(target), "--text", "no json here"]) == EXIT_VIOLATION
    assert main(["check", str(target), "--text", '{"steps": []} tool_run']) == EXIT_VIOLATION


def test_init_refuses_to_overwrite_without_force(tmp_path, capsys):
    target = tmp_path / "guard.md"
    target.write_text("# mine\n- MUST: keep me\n", encoding="utf-8")

    assert main(["init", str(target)]) == EXIT_ERROR
    assert "already exists" in capsys.readouterr().err
    assert "keep me" in target.read_text(encoding="utf-8")


def test_init_force_overwrites(tmp_path):
    target = tmp_path / "guard.md"
    target.write_text("# mine\n", encoding="utf-8")

    assert main(["init", str(target), "--force"]) == EXIT_OK
    assert "## Output format" in target.read_text(encoding="utf-8")


def test_init_name_option_sets_the_title(tmp_path):
    target = tmp_path / "whatever.md"
    assert main(["init", str(target), "--name", "reviewer"]) == EXIT_OK

    text = target.read_text(encoding="utf-8")
    assert text.startswith("# reviewer harness")
    assert Harness.from_file(target).name == "whatever"


def test_init_dash_prints_without_writing(tmp_path, capsys):
    assert main(["init", "-", "--name", "printed"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "# printed harness" in out
    assert "## Tone" in out
    assert not any(p.suffix == ".md" for p in tmp_path.iterdir())


def test_init_creates_missing_directories(tmp_path):
    target = tmp_path / "harnesses" / "nested" / "agent.md"
    assert main(["init", str(target)]) == EXIT_OK
    assert target.is_file()
