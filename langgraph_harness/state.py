"""State helpers: the per-node harness registry and recorded violations.

Inherit :class:`HarnessState` to add the two optional channels to a graph
state schema::

    from typing import Annotated
    from langgraph.graph.message import add_messages
    from langgraph_harness import HarnessState

    class State(HarnessState):
        messages: Annotated[list, add_messages]
"""

from __future__ import annotations

from typing import Any

try:  # Python 3.12+
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict


class HarnessState(TypedDict, total=False):
    """Mix-in channels used by :mod:`langgraph_harness`.

    Attributes:
        harness: Mapping of node name to the raw harness text that was bound.
        harness_violations: Violations recorded by nodes decorated with
            ``@harnessed(..., validate=True, on_violation="record")``.
    """

    harness: dict[str, str]
    harness_violations: list[dict[str, Any]]


def violations_of(state: Any) -> list[dict[str, Any]]:
    """Return the recorded violations from ``state`` (dict or object)."""
    if isinstance(state, dict):
        return list(state.get("harness_violations") or [])
    return list(getattr(state, "harness_violations", None) or [])


def harness_of(state: Any, node: str) -> str | None:
    """Return the raw harness text bound to ``node`` in ``state``."""
    if isinstance(state, dict):
        return (state.get("harness") or {}).get(node)
    harness = getattr(state, "harness", None) or {}
    return harness.get(node)


def with_harness(state: Any, node: str, harness_text: str) -> dict[str, Any]:
    """Return a copy of ``state`` with ``node`` mapped to ``harness_text``."""
    if isinstance(state, dict):
        current = dict(state.get("harness") or {})
        current[node] = harness_text
        output = dict(state)
        output["harness"] = current
        return output
    current = dict(getattr(state, "harness", None) or {})
    current[node] = harness_text
    return {"harness": current}
