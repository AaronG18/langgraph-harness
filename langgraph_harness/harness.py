"""Parse a Markdown HARNESS document into typed rules.

A harness is a small rules document, for example::

    # Planner harness

    ## Rules
    - MUST: respond with a JSON object [require=\\{\\s*"steps"]
    - MUST NOT: mention internal tool names [forbid=tool_]
    - SHOULD: keep the plan under 200 words

Rules are ``MUST``, ``MUST NOT`` or ``SHOULD``. The optional ``[require=...]``
and ``[forbid=...]`` annotations are regular expressions used by
:func:`langgraph_harness.validate_output` to check a node's output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import HarnessParseError

_RULE_RE = re.compile(r"^\s*[-*]\s*(MUST NOT|MUST|SHOULD)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)
_ANNOTATION_RE = re.compile(r"\[(require|forbid)\s*=\s*(.+?)\]", re.IGNORECASE | re.DOTALL)
# `#` is the document title; `##` and deeper group the rules that follow them.
_HEADING_RE = re.compile(r"^\s{0,3}(#{2,6})\s+(.+?)\s*#*\s*$")

_LEVELS = ("MUST", "MUST NOT", "SHOULD")


@dataclass(frozen=True, slots=True)
class Rule:
    """A single instruction line from a harness document."""

    id: str
    level: str
    text: str
    require: str | None = None
    forbid: str | None = None
    section: str = ""

    @property
    def is_hard(self) -> bool:
        """``MUST`` and ``MUST NOT`` are enforced; ``SHOULD`` is advisory."""
        return self.level in ("MUST", "MUST NOT")


def _label(rule: Rule) -> str:
    return f"({rule.section}) " if rule.section else ""


def _change_note(old: Rule, new: Rule) -> str:
    notes: list[str] = []
    if old.level != new.level:
        notes.append(f"{old.level} -> {new.level}")
    if old.require != new.require:
        notes.append("require changed")
    if old.forbid != new.forbid:
        notes.append("forbid changed")
    return ", ".join(notes) or "changed"


@dataclass(frozen=True, slots=True)
class HarnessDiff:
    """What changed between two harnesses, from the older one to the newer one."""

    added: tuple[Rule, ...] = ()
    removed: tuple[Rule, ...] = ()
    changed: tuple[tuple[Rule, Rule], ...] = ()

    @property
    def is_empty(self) -> bool:
        """``True`` when the two documents carry the same rules."""
        return not (self.added or self.removed or self.changed)

    def __len__(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)

    def format(self) -> str:
        """Render the diff as one line per change, or ``no changes``."""
        if self.is_empty:
            return "no changes"
        lines: list[str] = []
        for rule in self.added:
            lines.append(f"+ [{rule.level}] {_label(rule)}{rule.text}")
        for rule in self.removed:
            lines.append(f"- [{rule.level}] {_label(rule)}{rule.text}")
        for old, new in self.changed:
            lines.append(f"~ [{_change_note(old, new)}] {_label(new)}{new.text}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, list[dict[str, object]]]:
        return {
            "added": [{"section": r.section, "level": r.level, "text": r.text} for r in self.added],
            "removed": [{"section": r.section, "level": r.level, "text": r.text} for r in self.removed],
            "changed": [
                {
                    "section": new.section,
                    "text": new.text,
                    "old_level": old.level,
                    "new_level": new.level,
                }
                for old, new in self.changed
            ],
        }


@dataclass(frozen=True, slots=True)
class Harness:
    """A parsed harness document."""

    name: str
    raw: str
    rules: tuple[Rule, ...]

    @classmethod
    def from_text(cls, text: str, *, name: str = "harness") -> "Harness":
        """Parse ``text`` into a :class:`Harness`."""
        return parse_harness(text, name=name)

    @classmethod
    def from_file(cls, path: str | Path, *, name: str | None = None) -> "Harness":
        """Read and parse a harness file."""
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive
            raise HarnessParseError(f"cannot read harness file {p!s}: {exc}") from exc
        return parse_harness(text, name=name or p.stem)

    def rules_for(self, level: str) -> tuple[Rule, ...]:
        """Return every rule with the given level."""
        level = level.upper()
        if level not in _LEVELS:
            raise ValueError(f"unknown rule level {level!r}; expected one of {_LEVELS}")
        return tuple(rule for rule in self.rules if rule.level == level)

    def sections(self) -> dict[str, tuple[Rule, ...]]:
        """Group the rules by their ``##`` heading, in document order.

        Rules that appear before the first heading are collected under the empty
        string. A repeated heading name merges into the first occurrence's bucket.
        """
        grouped: dict[str, list[Rule]] = {}
        for rule in self.rules:
            grouped.setdefault(rule.section, []).append(rule)
        return {name: tuple(rules) for name, rules in grouped.items()}

    def rules_in(self, section: str) -> tuple[Rule, ...]:
        """Return the rules of one section; ``""`` means before the first heading."""
        return tuple(rule for rule in self.rules if rule.section == section)

    def diff(self, other: "Harness") -> HarnessDiff:
        """Return the changes needed to turn this harness into ``other``.

        Rules are matched on ``(section, text)``, case-insensitively; repeated copies of
        the same text are paired in order. Consequences worth knowing: a rule that only
        moved to another section appears as one removal plus one addition, and editing the
        text of a rule also shows up as remove + add rather than as a change.
        """
        remaining: dict[tuple[str, str], list[Rule]] = {}
        for rule in self.rules:
            key = (rule.section.strip().lower(), rule.text.strip().lower())
            remaining.setdefault(key, []).append(rule)

        added: list[Rule] = []
        changed: list[tuple[Rule, Rule]] = []
        for rule in other.rules:
            key = (rule.section.strip().lower(), rule.text.strip().lower())
            bucket = remaining.get(key)
            if bucket:
                old = bucket.pop(0)
                if (old.level, old.require, old.forbid) != (rule.level, rule.require, rule.forbid):
                    changed.append((old, rule))
            else:
                added.append(rule)

        removed = [rule for bucket in remaining.values() for rule in bucket]
        return HarnessDiff(tuple(added), tuple(removed), tuple(changed))

    def prompt(self) -> str:
        """Render the harness as a compact system-prompt block.

        A document without ``##`` headings renders as a flat list, exactly as
        before sections existed. A document with headings gains a ``## name``
        line per section so the model can tell the groups apart.
        """
        lines = [f"HARNESS: {self.name}", "You must obey these rules for this node:"]
        if not self.rules:
            lines.append("- (no machine-readable rules found)")
            return "\n".join(lines)

        if all(not rule.section for rule in self.rules):
            for rule in self.rules:
                lines.append(f"- [{rule.level}] {rule.text}")
            return "\n".join(lines)

        for name, rules in self.sections().items():
            if name:
                lines.append(f"## {name}")
            for rule in rules:
                lines.append(f"- [{rule.level}] {rule.text}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.rules)


def parse_harness(text: str, *, name: str = "harness") -> Harness:
    """Parse Markdown ``text`` into a :class:`Harness`.

    Raises:
        HarnessParseError: if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise HarnessParseError(f"harness source must be str, got {type(text).__name__}")

    rules: list[Rule] = []
    section = ""
    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading is not None:
            section = heading.group(2).strip()
            continue

        match = _RULE_RE.match(line)
        if match is None:
            continue

        level = " ".join(match.group(1).upper().split())
        body = match.group(2).strip()

        require: str | None = None
        forbid: str | None = None
        for annotation in _ANNOTATION_RE.finditer(body):
            kind = annotation.group(1).lower()
            value = annotation.group(2).strip()
            if kind == "require":
                require = value
            else:
                forbid = value

        rules.append(
            Rule(
                id=f"R{len(rules) + 1}",
                level=level,
                text=_ANNOTATION_RE.sub("", body).strip(),
                require=require,
                forbid=forbid,
                section=section,
            )
        )

    return Harness(name=name, raw=text, rules=tuple(rules))


HarnessLike = "str | Path | Harness"


def load_harness(source: str | Path | Harness, *, name: str | None = None) -> Harness:
    """Resolve ``source`` (a path, a raw document or a :class:`Harness`)."""
    if isinstance(source, Harness):
        return source
    if isinstance(source, Path):
        return Harness.from_file(source, name=name)

    if not isinstance(source, str):
        raise HarnessParseError(f"harness source must be str, Path or Harness, got {type(source).__name__}")

    # A short, single-line string that names an existing file is treated as a path.
    if "\n" not in source and len(source) < 260:
        try:
            path = Path(source)
            if path.is_file():
                return Harness.from_file(path, name=name)
        except (OSError, ValueError):
            pass

    return parse_harness(source, name=name or "harness")
