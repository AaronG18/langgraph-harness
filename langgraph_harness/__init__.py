"""langgraph-harness — per-node HARNESS parsing, injection and enforcement for LangGraph.

LangGraph nodes are plain callables: ``StateGraph.add_node`` accepts metadata,
retry/cache policies and an input schema, but it has no notion of a per-node
instruction document. ``create_react_agent`` has a single graph-level prompt.

This package adds the missing piece: bind a small Markdown HARNESS document to a
node, expose it inside the node body, optionally validate the node's output, and
optionally gate an edge on violations.

Quick start::

    from langgraph_harness import Harness, harnessed, current_harness, harness_messages

    PLANNER = Harness.from_text('''
    - MUST: return a JSON object [require="steps"]
    - MUST NOT: mention internal tool names [forbid=tool_]
    ''', name="planner")

    @harnessed(PLANNER, validate=True)
    def planner(state, config=None):
        messages = [*harness_messages(), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}
"""

from .context import current_harness, reset_current_harness, set_current_harness
from .decorators import harnessed
from .edges import harness_gate
from .errors import HarnessError, HarnessNotBound, HarnessParseError, HarnessViolation
from .harness import Harness, Rule, load_harness, parse_harness
from .prompts import harness_messages, harness_prompt
from .state import HarnessState, harness_of, violations_of, with_harness
from .validation import Report, Violation, validate_output

__version__ = "0.1.2"

__all__ = [
    # parsing
    "Harness",
    "Rule",
    "load_harness",
    "parse_harness",
    # node decorator (primary form)
    "harnessed",
    "current_harness",
    "set_current_harness",
    "reset_current_harness",
    "harness_messages",
    "harness_prompt",
    # state form
    "HarnessState",
    "harness_of",
    "violations_of",
    "with_harness",
    # edge form
    "harness_gate",
    # validation
    "validate_output",
    "Report",
    "Violation",
    # errors
    "HarnessError",
    "HarnessParseError",
    "HarnessNotBound",
    "HarnessViolation",
    "__version__",
]
