"""Check a node's output against the harness rules.

Only the ``require`` / ``forbid`` annotations are machine-checkable; a rule
without an annotation is still injected into the prompt but cannot be verified
automatically. ``MUST`` and ``MUST NOT`` failures are violations; ``SHOULD``
failures are warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .harness import Harness, Rule


@dataclass(frozen=True, slots=True)
class Violation:
    """A single failed rule."""

    rule_id: str
    level: str
    message: str
    rule_text: str

    def as_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "message": self.message,
            "rule_text": self.rule_text,
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
        return Violation(rule.id, rule.level, f"required pattern not found: {rule.require}", rule.text)
    if rule.forbid is not None and re.search(rule.forbid, text, re.IGNORECASE | re.DOTALL) is not None:
        return Violation(rule.id, rule.level, f"forbidden pattern found: {rule.forbid}", rule.text)
    return None


def validate_output(text: str, harness: Harness) -> Report:
    """Validate ``text`` against ``harness`` and return a :class:`Report`."""
    if not isinstance(text, str):
        raise TypeError(f"validate_output() expects str, got {type(text).__name__}")

    violations: list[Violation] = []
    warnings: list[Violation] = []

    for rule in harness.rules:
        failure = _check_rule(rule, text)
        if failure is None:
            continue
        if rule.is_hard:
            violations.append(failure)
        else:
            warnings.append(failure)

    return Report(tuple(violations), tuple(warnings))
