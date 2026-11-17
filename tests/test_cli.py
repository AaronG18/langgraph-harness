"""The CLI is the CI-facing surface: exit codes and JSON must be stable."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from langgraph_harness.cli import EXIT_ERROR, EXIT_OK, EXIT_VIOLATION, main

HARNESS = """# guard
- MUST: answer SAFE [require=SAFE]
- MUST NOT: mention SECRET [forbid=SECRET]
- SHOULD: stay short
"""


@pytest.fixture()
def harness_file(tmp_path):
    path = tmp_path / "guard.md"
    path.write_text(HARNESS, encoding="utf-8")
    return path


def test_check_passes_and_exits_zero(harness_file, tmp_path, capsys):
    out = tmp_path / "out.txt"
    out.write_text("SAFE answer", encoding="utf-8")

    assert main(["check", str(harness_file), str(out)]) == EXIT_OK
    assert "OK" in capsys.readouterr().out


def test_check_reports_violation_and_exits_one(harness_file, tmp_path, capsys):
    out = tmp_path / "out.txt"
    out.write_text("SECRET leaked", encoding="utf-8")

    assert main(["check", str(harness_file), str(out)]) == EXIT_VIOLATION
    printed = capsys.readouterr().out
    assert "VIOLATION R2" in printed
    assert "R1" in printed


def test_check_json_is_machine_readable(harness_file, tmp_path, capsys):
    out = tmp_path / "out.txt"
    out.write_text("SECRET", encoding="utf-8")

    assert main(["check", str(harness_file), str(out), "--json"]) == EXIT_VIOLATION
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["harness"] == "guard"
    assert {v["rule_id"] for v in payload["violations"]} == {"R1", "R2"}


def test_check_reads_inline_text(harness_file, capsys):
    assert main(["check", str(harness_file), "--text", "SAFE"]) == EXIT_OK
    assert "OK" in capsys.readouterr().out


def test_check_reads_stdin(harness_file, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("SECRET"))
    assert main(["check", str(harness_file)]) == EXIT_VIOLATION
    assert "VIOLATION" in capsys.readouterr().out


def test_check_dash_means_stdin(harness_file, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("SAFE"))
    assert main(["check", str(harness_file), "-"]) == EXIT_OK
    assert "OK" in capsys.readouterr().out


def test_strict_promotes_warning_to_failure(tmp_path, capsys):
    harness = tmp_path / "should.md"
    harness.write_text("- SHOULD: mention SAFE [require=SAFE]\n", encoding="utf-8")

    assert main(["check", str(harness), "--text", "other"]) == EXIT_OK
    warns = capsys.readouterr().out
    assert "WARNING R1" in warns

    assert main(["check", str(harness), "--text", "other", "--strict"]) == EXIT_VIOLATION
    strict = capsys.readouterr().out
    assert "VIOLATION R1" in strict


def test_missing_harness_file_exits_two(tmp_path, capsys):
    missing = tmp_path / "nope.md"
    assert main(["check", str(missing), "--text", "x"]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_prompt_prints_the_rules(harness_file, capsys):
    assert main(["prompt", str(harness_file)]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "HARNESS: guard" in printed
    assert "[MUST] answer SAFE" in printed


def test_module_entry_point_runs(tmp_path):
    harness = tmp_path / "guard.md"
    harness.write_text(HARNESS, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "langgraph_harness", "check", str(harness), "--text", "SAFE"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "OK" in proc.stdout
