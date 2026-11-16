"""Optional EDGE form: route to a repair node when a node violated its harness."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END

from .state import violations_of


def harness_gate(
    *,
    on_violation: str,
    otherwise: Any = END,
) -> Callable[[Any], Any]:
    """Build a conditional-edge router.

    Use it with ``add_conditional_edges(node, harness_gate(on_violation="repair"))``.
    The router returns ``on_violation`` when ``state["harness_violations"]`` is
    non-empty, else ``otherwise`` (``END`` by default).

    Args:
        on_violation: Name of the repair / fallback node to route to.
        otherwise: Destination when there are no violations (default: ``END``).
    """

    def route(state: Any) -> Any:
        return on_violation if violations_of(state) else otherwise

    return route
