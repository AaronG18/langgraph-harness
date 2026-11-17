"""The ``@harnessed`` node decorator — the primary form of this package."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from .context import reset_current_harness, set_current_harness
from .errors import HarnessViolation
from .harness import HarnessLike, load_harness
from .prompts import harness_messages
from .state import repair_count_of
from .validation import validate_output

F = TypeVar("F", bound=Callable[..., Any])

_ON_VIOLATION = ("raise", "record")


def harnessed(
    harness: HarnessLike,
    *,
    validate: bool = False,
    on_violation: str = "raise",
    inject_messages: bool = False,
    count_repairs: bool = False,
    name: str | None = None,
) -> Callable[[F], F]:
    """Bind a HARNESS to a LangGraph node.

    Args:
        harness: A :class:`~langgraph_harness.Harness`, a Markdown string, or a
            path to a Markdown file.
        validate: When ``True``, the node's returned text is checked against the
            harness rules.
        on_violation: ``"raise"`` (default) raises
            :class:`~langgraph_harness.HarnessViolation`; ``"record"`` appends the
            violations to ``harness_violations`` in the returned state dict.
        inject_messages: When ``True`` and the node's first argument is a state
            dict containing ``messages``, a ``SystemMessage`` with the rendered
            harness is prepended to a copy of those messages.
        count_repairs: When ``True`` and ``on_violation="record"``, also increment
            ``harness_repair_count`` in the returned state. Combine it with
            ``harness_gate(max_retries=...)`` so a repair loop stops with a
            destination of your choice instead of hitting ``recursion_limit``.
        name: Node name used in error messages (defaults to the function name).

    Returns:
        The decorated node. The original function is available as
        ``wrapper.__wrapped__``.
    """
    parsed = load_harness(harness, name=name)
    if on_violation not in _ON_VIOLATION:
        raise ValueError(f"on_violation must be one of {_ON_VIOLATION}, got {on_violation!r}")

    def decorator(fn: F) -> F:
        node_name = name or getattr(fn, "__name__", "node")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = set_current_harness(parsed)
                try:
                    call_args = _maybe_inject(args, inject_messages)
                    result = await fn(*call_args, **kwargs)
                finally:
                    reset_current_harness(token)
                return _maybe_validate(
                    result,
                    parsed,
                    node_name,
                    validate,
                    on_violation,
                    state=args[0] if args else None,
                    count_repairs=count_repairs,
                )

            wrapper: Any = async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = set_current_harness(parsed)
                try:
                    call_args = _maybe_inject(args, inject_messages)
                    result = fn(*call_args, **kwargs)
                finally:
                    reset_current_harness(token)
                return _maybe_validate(
                    result,
                    parsed,
                    node_name,
                    validate,
                    on_violation,
                    state=args[0] if args else None,
                    count_repairs=count_repairs,
                )

            wrapper = sync_wrapper

        wrapper.__harness__ = parsed
        wrapper.__harness_name__ = parsed.name
        wrapper.__harness_node__ = node_name
        return wrapper

    return decorator


def _maybe_inject(args: tuple[Any, ...], enabled: bool) -> tuple[Any, ...]:
    if not enabled or not args:
        return args
    state = args[0]
    if isinstance(state, dict) and isinstance(state.get("messages"), list):
        injected = dict(state)
        injected["messages"] = [*harness_messages(), *state["messages"]]
        return (injected, *args[1:])
    return args


def _extract_text(result: Any) -> str | None:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            last = messages[-1]
            # Plain strings are accepted so the decorator also works with simple
            # dict states that do not use LangChain message objects.
            if isinstance(last, str):
                return last
            content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(part) for part in content)
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    return None


def _maybe_validate(
    result: Any,
    parsed: Any,
    node_name: str,
    validate: bool,
    on_violation: str,
    *,
    state: Any = None,
    count_repairs: bool = False,
) -> Any:
    del node_name  # kept for a clearer signature / future messages
    if not validate:
        return result

    text = _extract_text(result)
    if text is None:
        return result

    report = validate_output(text, parsed)
    if report.ok:
        # The node now complies. Clear the violations that a previous attempt left
        # in state; otherwise harness_gate() would keep seeing them and route back
        # to the repair node forever.
        if isinstance(result, dict) and isinstance(state, dict) and state.get("harness_violations"):
            output = dict(result)
            output["harness_violations"] = []
            return output
        return result

    if on_violation == "raise":
        raise HarnessViolation(parsed.name, report)

    if isinstance(result, dict):
        output = dict(result)
        # "record" means "these are the violations of THIS attempt": overwrite rather
        # than append, so a later compliant attempt can clear them.
        output["harness_violations"] = report.as_dicts()
        if count_repairs:
            history = state if isinstance(state, dict) else output
            output["harness_repair_count"] = repair_count_of(history) + 1
        return output

    raise HarnessViolation(parsed.name, report)
