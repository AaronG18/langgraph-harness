"""Loading a folder of harness files, the shape a real project keeps them in."""

from __future__ import annotations

import pytest

from langgraph_harness import Harness, HarnessParseError, load_harnesses


def _write(path, rules="- MUST: answer SAFE [require=SAFE]\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {path.stem}\n\n{rules}", encoding="utf-8")


def test_flat_directory_keys_are_stems(tmp_path):
    _write(tmp_path / "planner.md")
    _write(tmp_path / "writer.md", "- MUST: answer WRITER [require=WRITER]\n")

    loaded = load_harnesses(tmp_path)

    assert list(loaded) == ["planner", "writer"]
    assert all(isinstance(v, Harness) for v in loaded.values())
    assert loaded["planner"].name == "planner"
    assert len(loaded["writer"]) == 1


def test_nested_files_keep_their_relative_key(tmp_path):
    _write(tmp_path / "agents" / "planner.md")
    _write(tmp_path / "agents" / "deep" / "reviewer.md")

    loaded = load_harnesses(tmp_path)

    assert list(loaded) == ["agents/deep/reviewer", "agents/planner"]
    assert loaded["agents/deep/reviewer"].name == "agents/deep/reviewer"


def test_two_files_with_the_same_stem_do_not_collide(tmp_path):
    _write(tmp_path / "a" / "planner.md")
    _write(tmp_path / "b" / "planner.md", "- MUST: other [require=other]\n")

    loaded = load_harnesses(tmp_path)

    assert sorted(loaded) == ["a/planner", "b/planner"]
    assert loaded["a/planner"].rules[0].require == "SAFE"
    assert loaded["b/planner"].rules[0].require == "other"


def test_non_markdown_files_are_ignored(tmp_path):
    _write(tmp_path / "planner.md")
    (tmp_path / "notes.txt").write_text("- MUST: not a harness\n", encoding="utf-8")

    assert list(load_harnesses(tmp_path)) == ["planner"]


def test_custom_pattern(tmp_path):
    _write(tmp_path / "one.harness")
    _write(tmp_path / "two.md")

    assert list(load_harnesses(tmp_path, pattern="*.harness")) == ["one"]


def test_recursive_can_be_turned_off(tmp_path):
    _write(tmp_path / "top.md")
    _write(tmp_path / "nested" / "inner.md")

    assert list(load_harnesses(tmp_path, recursive=False)) == ["top"]
    assert list(load_harnesses(tmp_path)) == ["nested/inner", "top"]


def test_a_single_file_yields_a_one_entry_mapping(tmp_path):
    target = tmp_path / "planner.md"
    _write(target)

    loaded = load_harnesses(target)
    assert list(loaded) == ["planner"]


def test_missing_path_raises(tmp_path):
    with pytest.raises(HarnessParseError):
        load_harnesses(tmp_path / "nope")


def test_empty_directory_returns_empty_mapping(tmp_path):
    assert load_harnesses(tmp_path) == {}
