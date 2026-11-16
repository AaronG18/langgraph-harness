"""Exceptions raised by :mod:`langgraph_harness`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .validation import Report


class HarnessError(Exception):
    """Base class for every error raised by this package."""


class HarnessParseError(HarnessError):
    """Raised when a harness document cannot be parsed."""


class HarnessNotBound(HarnessError):
    """Raised when current_harness() is called outside a harnessed node."""


class HarnessViolation(HarnessError):
    """Raised when a node's output violates a MUST / MUST NOT rule."""

    def __init__(self, harness_name: str, report: "Report") -> None:
        self.harness_name = harness_name
        self.report = report
        details = "; ".join(f"{v.rule_id} [{v.level}] {v.message}" for v in report.violations)
        super().__init__(f"node violated harness {harness_name!r}: {details}")
