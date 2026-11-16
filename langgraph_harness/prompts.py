"""Helpers that turn a harness into prompt messages."""

from __future__ import annotations

from typing import Any

from .context import current_harness
from .harness import Harness


def harness_prompt(harness: Harness | None = None) -> str:
    """Return the rendered prompt block for ``harness`` (default: current)."""
    return (harness or current_harness()).prompt()


def harness_messages(harness: Harness | None = None) -> list[Any]:
    """Return ``[SystemMessage(content=<harness prompt>)]``.

    ``langchain-core`` is imported lazily so the rest of the package works
    without LangChain installed.
    """
    try:
        from langchain_core.messages import SystemMessage
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "harness_messages() requires langchain-core, which ships with langgraph. "
            "Install it with `pip install langgraph`."
        ) from exc

    return [SystemMessage(content=harness_prompt(harness))]
