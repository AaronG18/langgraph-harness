"""Per-node harness binding through a :class:`contextvars.ContextVar`.

``@harnessed`` sets the current harness for the duration of the node call, so
the node body can call :func:`current_harness` without threading the document
through its arguments. Context variables are task-local, so concurrent node
executions do not see each other's harness.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

from .errors import HarnessNotBound
from .harness import Harness

_CURRENT: ContextVar[Harness | None] = ContextVar("langgraph_harness_current", default=None)


def current_harness() -> Harness:
    """Return the harness bound to the currently executing node.

    Raises:
        HarnessNotBound: if called outside a ``@harnessed`` node.
    """
    harness = _CURRENT.get()
    if harness is None:
        raise HarnessNotBound(
            "current_harness() was called outside a @harnessed node. "
            "Decorate the node with @harnessed(...) first."
        )
    return harness


def set_current_harness(harness: Harness) -> Token[Harness | None]:
    """Bind ``harness`` to the current context and return a reset token."""
    return _CURRENT.set(harness)


def reset_current_harness(token: Token[Harness | None]) -> None:
    """Restore the previous harness binding."""
    _CURRENT.reset(token)
