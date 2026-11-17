"""Check a node's output against the harness rules.

Only the ``require`` / ``forbid`` annotations are machine-checkable; a rule
without an annotation is still injected into the prompt but cannot be verified
automatically. ``MUST`` and ``MUST NOT`` failures are violations; ``SHOULD``
failures are warnings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .harness import Harness, Rule


@dataclass(frozen=True, slots=True)
class Violation:
    """A single failed rule."""

    rule_id: str
    level: str
    message: str
    rule_text: str
    section: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "message": self.message,
            "rule_text": self.rule_text,
            "section": self.section,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """Result of validating one node output."""

    violations: tuple[Violation, ...] = ()
    warnings: tuple[Violation, ...] = ()

    @property
    def ok(self) -> bool:
        """``True`` when no hard rule failed."""
        return not self.violations

    def as_dicts(self) -> list[dict[str, str]]:
        return [v.as_dict() for v in self.violations]


def _check_rule(rule: Rule, text: str) -> Violation | None:
    if rule.require is not None and re.search(rule.require, text, re.IGNORECASE | re.DOTALL) is None:
        return Violation(rule.id, rule.level, f"required pattern not found: {rule.require}", rule.text, rule.section)
    if rule.forbid is not None and re.search(rule.forbid, text, re.IGNORECASE | re.DOTALL) is not None:
        return Violation(rule.id, rule.level, f"forbidden pattern found: {rule.forbid}", rule.text, rule.section)
    return None


def validate_output(text: str, harness: Harness, *, sections: Iterable[str] | None = None) -> Report:
    """Validate ``text`` against ``harness`` and return a :class:`Report`.

    Args:
        text: The node output to check.
        harness: The parsed harness.
        sections: When given, only rules belonging to these ``##`` sections are
            checked (``""`` selects the rules written before the first heading).
    """
    if not isinstance(text, str):
        raise TypeError(f"validate_output() expects str, got {type(text).__name__}")

    selected = None if sections is None else set(sections)
    violations: list[Violation] = []
    warnings: list[Violation] = []

    for rule in harness.rules:
        if selected is not None and rule.section not in selected:
            continue
        failure = _check_rule(rule, text)
        if failure is None:
            continue
        if rule.is_hard:
            violations.append(failure)
        else:
            warnings.append(failure)

    return Report(tuple(violations), tuple(warnings))
