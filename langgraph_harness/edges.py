"""Optional EDGE form: route to a repair node when a node violated its harness."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END

from .state import repair_count_of, violations_of


def harness_gate(
    *,
    on_violation: str,
    otherwise: Any = END,
    max_retries: int | None = None,
    on_exhausted: Any = END,
    counter_key: str = "harness_repair_count",
) -> Callable[[Any], Any]:
    """Build a conditional-edge router.

    Use it with ``add_conditional_edges(node, harness_gate(on_violation="repair"))``.
    The router returns ``on_violation`` when ``state["harness_violations"]`` is
    non-empty, else ``otherwise`` (``END`` by default).

    Without ``max_retries`` the router keeps sending the graph to the repair node
    for as long as violations exist, which can loop until LangGraph raises
    ``GraphRecursionError``. Set ``max_retries`` to bound that loop: after that many
    repair attempts the router goes to ``on_exhausted`` instead.

    Args:
        on_violation: Name of the repair / fallback node to route to.
        otherwise: Destination when there are no violations (default: ``END``).
        max_retries: Maximum number of repair attempts before giving up. ``None``
            (default) keeps the previous unbounded behaviour.
        on_exhausted: Destination once ``max_retries`` is reached (default: ``END``).
        counter_key: State key that counts repair attempts. The counter is
            incremented by ``@harnessed(..., count_repairs=True)`` or by
            :func:`langgraph_harness.record_repair_attempt`.
    """
    if max_retries is not None and max_retries < 0:
        raise ValueError(f"max_retries must be >= 0 or None, got {max_retries!r}")

    def route(state: Any) -> Any:
        if not violations_of(state):
            return otherwise
        if max_retries is None:
            return on_violation
        if repair_count_of(state, counter_key=counter_key) < max_retries:
            return on_violation
        return on_exhausted

    return route
