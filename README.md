# langgraph-harness

**Per-node HARNESS parsing, injection and enforcement for LangGraph nodes.**

A [LangGraph](https://github.com/langchain-ai/langgraph) graph is a set of plain
callables. `StateGraph.add_node` accepts metadata, retry/cache policies, an input
schema, error handlers and timeouts — but it has **no notion of a per-node
instruction document**. `create_react_agent` has a single **graph-level** `prompt`
(plus `pre_model_hook` / `post_model_hook` for global interception). So today the
answer to *"can each node's agent parse its own HARNESS?"* is: **not natively.**

`langgraph-harness` adds that missing layer:

```python
from langgraph_harness import Harness, harnessed, current_harness, harness_messages

PLANNER = Harness.from_text("""
- MUST: return a JSON object [require="steps"]
- MUST NOT: mention internal tool names [forbid=tool_]
- SHOULD: keep the plan under 200 words
""", name="planner")

@harnessed(PLANNER, validate=True)
def planner(state, config=None):
    messages = [*harness_messages(), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

`current_harness()` inside the node returns *that node's* harness — not a global
prompt, and not another node's document.

## Why a node decorator (and not only an edge or a state field)?

| Form | What it can do | Why it is not enough alone |
| --- | --- | --- |
| **Node decorator** | Binds the document to one node, exposes it inside the node, validates the output | — (this is the primary form) |
| **State field** | Carries the node→harness registry and violations across steps | Every node still has to read it manually |
| **Edge** | Routes to a repair/fallback node when rules fail | An edge runs *after* the node; it cannot inject instructions into the node's LLM call |

The package therefore ships all three, with the decorator as the entry point:
`@harnessed(...)` binds and injects, `HarnessState` records, and
`harness_gate(...)` routes.

## Install

```bash
pip install langgraph-harness
```

On PyPI: <https://pypi.org/project/langgraph-harness/>.

Requires Python 3.10+ and `langgraph>=1.0`.

## The HARNESS format

A harness is a small Markdown document. Only bullet lines that start with `MUST`,
`MUST NOT` or `SHOULD` are parsed as rules:

```markdown
# Planner harness

- MUST: return a JSON object [require="steps"]
- MUST NOT: mention internal tool names [forbid=tool_]
- SHOULD: keep the plan short
```

* `MUST` / `MUST NOT` are **enforced**; `SHOULD` failures are **warnings**.
* `[require=REGEX]` — the output must match the regex.
* `[forbid=REGEX]` — the output must not match the regex.
* Rules without an annotation are still injected into the prompt; they simply
  cannot be verified automatically.

### Sections

`#` is the document title; a `##` (or deeper) heading groups the rules below it.
The prompt then names each group, and validation can be scoped to one of them:

```markdown
# Planner harness

- MUST: reply in English [require=[A-Za-z]]

## Output format
- MUST: return JSON [require=\{\s*"steps"]

## Tone
- MUST NOT: apologise [forbid=^\s*I'm sorry]
```

```python
harness.sections()                      # {"": (R1,), "Output format": (R2,), "Tone": (R3,)}
harness.rules_in("Tone")                # (R3,)
validate_output(text, harness, sections=["Output format"])   # only R2 is checked
```

A document without `##` headings keeps the flat prompt it always had, and
`violations`/`warnings` now carry the `section` they came from.

## Three forms

### 1. Node decorator (primary)

```python
from langgraph_harness import harnessed, current_harness

@harnessed("harnesses/planner.md", validate=True)
def planner(state):
    harness = current_harness()          # this node's document
    ...
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `validate` | `False` | Check the node's returned text against `require` / `forbid` |
| `on_violation` | `"raise"` | `"raise"` → `HarnessViolation`; `"record"` → append to `harness_violations` |
| `inject_messages` | `False` | Prepend a `SystemMessage` with the harness to a copy of `state["messages"]` |
| `name` | function name | Node name used in errors |
| `count_repairs` | `False` | On `"record"`, also increment `harness_repair_count` |

The node may return a string, a state dict, **or a `Command`**. For a `Command` the
text inside `update` is validated, and the recorded violations are merged into that
same `update`, so `goto` / `graph` / `resume` come through untouched:

```python
@harnessed(PLANNER, validate=True, on_violation="record", count_repairs=True)
def planner(state):
    return Command(update={"messages": [answer]}, goto="tools")
```

### 2. State

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph_harness import HarnessState

class State(HarnessState):
    messages: Annotated[list, add_messages]
```

`HarnessState` adds three optional channels: `harness` (node → raw text),
`harness_violations` (violations of the latest attempt, cleared once a node complies)
and `harness_repair_count` (how many repair attempts have run). Helpers: `harness_of`,
`violations_of`, `with_harness`, `repair_count_of`, `record_repair_attempt`.

### 3. Edge gate

```python
from langgraph.graph import END, START, StateGraph
from langgraph_harness import harness_gate

builder.add_conditional_edges(
    "planner",
    harness_gate(on_violation="repair", otherwise=END),
)
```

When `state["harness_violations"]` is non-empty the router returns `"repair"`.
Only `MUST` / `MUST NOT` failures are recorded, so a `SHOULD` warning never sends the
graph down the repair branch.

A gate can also listen to one section only, which is what you want when a repair node
can fix formatting but not tone:

```python
builder.add_conditional_edges(
    "planner",
    harness_gate(on_violation="repair", sections=["Output format"]),
)
```

A violation recorded under `Tone` is then ignored by this edge, and a violation
recorded before the first `##` heading belongs to the empty section name `""`.

Without a bound, a repair node that keeps failing sends the graph back forever and
ends in `GraphRecursionError`. Set `max_retries` to cap it:

```python
@harnessed(PLANNER, validate=True, on_violation="record", count_repairs=True)
def planner(state): ...

builder.add_conditional_edges(
    "planner",
    harness_gate(on_violation="repair", max_retries=2, on_exhausted="give_up"),
)
```

`count_repairs=True` increments `harness_repair_count` on each violation; after
`max_retries` attempts the router goes to `on_exhausted` instead of looping. A node
that finally complies clears `harness_violations`, so the gate returns `otherwise`.

## API at a glance

```python
Harness.from_text(text, name=...)      # parse Markdown
Harness.from_file(path, name=...)      # parse a file
parse_harness(text, name=...)          # same, functional form
load_harness(str | Path | Harness)     # resolve any of the three

harnessed(harness, *, validate, on_violation, inject_messages, name)
current_harness()                      # inside a harnessed node
harness_messages() / harness_prompt()  # render to a SystemMessage / str

HarnessState                           # TypedDict mix-in
violations_of(state, sections=None) / harness_of(state, node) / with_harness(state, node, text)
repair_count_of(state) / record_repair_attempt(state)

harness_gate(on_violation="repair", otherwise=END,
             max_retries=None, on_exhausted=END, sections=None)

harness.sections() / harness.rules_in(section)
harness.diff(other) -> HarnessDiff(added, removed, changed)   # .format() / .as_dict()

validate_output(text, harness, sections=None) -> Report(violations, warnings)
```

## Command line

The same checks are available without writing Python, which is what a CI job
needs when it has a fixture and the harness that produced it:

```bash
# exit 0 pass, 1 violation, 2 usage/parse error
langgraph-harness check guard.md output.txt
langgraph-harness check guard.md --text "SAFE answer"
cat output.txt | langgraph-harness check guard.md

langgraph-harness check guard.md output.txt --json     # machine-readable
langgraph-harness check guard.md output.txt --strict   # SHOULD warns fail too
langgraph-harness check guard.md output.txt --section "Output format"
langgraph-harness prompt guard.md                      # what the node receives
langgraph-harness init harnesses/planner.md            # starter document with sections
langgraph-harness diff old.md new.md                   # 0 identical / 1 different / 2 error
```

`python -m langgraph_harness ...` is equivalent when the console script is not on
`PATH`. The JSON form is stable: `{harness, ok, violations[], warnings[]}`.

## How it works

`@harnessed` binds the parsed harness to a `contextvars.ContextVar` for the
duration of the call and resets it in a `finally` block. Context variables are
task-local, so concurrent nodes (sync or async) do not see each other's harness.
The decorated wrapper also exposes `__harness__`, `__harness_name__` and
`__harness_node__` for introspection.

## Verification

```bash
pytest -q
# 88 passed
```

The integration tests build a real `StateGraph` with two nodes that each call
`current_harness()` and assert they see *only their own* document, and a graph
where a violating node is routed to a repair node by `harness_gate`.

## Real-model demo

`examples/real_llm_demo.py` runs two nodes with conflicting harnesses and
asserts that each node answers with its own word:

```bash
python examples/real_llm_demo.py --mock                  # deterministic, no key
OPENAI_API_KEY=...   python examples/real_llm_demo.py --provider openai
DEEPSEEK_API_KEY=... python examples/real_llm_demo.py --provider deepseek
GEMINI_API_KEY=...   python examples/real_llm_demo.py --provider gemini
OLLAMA_MODEL=qwen2.5:1.5b python examples/real_llm_demo.py --provider ollama
```

## Publishing

* `.github/workflows/ci.yml` runs the test suite on Python 3.10–3.13.
* `.github/workflows/publish.yml` builds the sdist/wheel, runs `twine check`,
  and uploads to PyPI on a GitHub release. It works with a `PYPI_API_TOKEN`
  repository secret or with PyPI trusted publishing (no secret).

Locally:

```bash
python -m build
twine check dist/*
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-... twine upload dist/*
```

## Limitations

* Regex checks are shallow: they prove a string contains (or avoids) a pattern,
  not that the model *semantically* followed the rule. Use the harness as a
  guardrail, not as a proof.
* `inject_messages=True` prepends a `SystemMessage` to the node's copy of
  `state["messages"]`; the node owns the LLM call and should not echo that
  message back into state.
* The package does not call an LLM. It parses, injects, validates and routes;
  the model invocation stays in your node.

## License

MIT
