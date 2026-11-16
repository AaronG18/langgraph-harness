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

_LEVELS = ("MUST", "MUST NOT", "SHOULD")


@dataclass(frozen=True, slots=True)
class Rule:
    """A single instruction line from a harness document."""

    id: str
    level: str
    text: str
    require: str | None = None
    forbid: str | None = None

    @property
    def is_hard(self) -> bool:
        """``MUST`` and ``MUST NOT`` are enforced; ``SHOULD`` is advisory."""
        return self.level in ("MUST", "MUST NOT")


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

    def prompt(self) -> str:
        """Render the harness as a compact system-prompt block."""
        lines = [f"HARNESS: {self.name}", "Follow these rules for this node:"]
        if not self.rules:
            lines.append("- (no machine-readable rules found)")
        for rule in self.rules:
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
    for line in text.splitlines():
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
