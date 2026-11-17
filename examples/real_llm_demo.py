"""End-to-end demo: two LangGraph nodes, each using its own HARNESS.

The point is that each node sends *its own* system prompt to the model. A
compliant model therefore answers "PLANNER" at the planner node and "WRITER" at
the writer node, even though the user question is identical.

Run without credentials (deterministic fake model):

    python examples/real_llm_demo.py --mock

Run against a real provider:

    OPENAI_API_KEY=...     python examples/real_llm_demo.py --provider openai
    DEEPSEEK_API_KEY=...   python examples/real_llm_demo.py --provider deepseek
    GEMINI_API_KEY=...     python examples/real_llm_demo.py --provider gemini
    python examples/real_llm_demo.py --provider ollama   # local Ollama

`--provider auto` (the default) picks the first configured provider, falling
back to `mock`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import httpx
from langgraph.graph import END, START, StateGraph

from langgraph_harness import (
    Harness,
    HarnessState,
    current_harness,
    harness_prompt,
    harnessed,
)

PLANNER = Harness.from_text(
    """
# Planner
- MUST: reply with exactly the single word PLANNER and nothing else [require=PLANNER]
""",
    name="planner",
)

WRITER = Harness.from_text(
    """
# Writer
- MUST: reply with exactly the single word WRITER and nothing else [require=WRITER]
""",
    name="writer",
)

QUESTION = "Reply with exactly one word."


def _chat_completions(base_url: str, api_key: str, model: str) -> Callable[[str, str], str]:
    def chat(system: str, user: str) -> str:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    return chat


def _gemini(api_key: str, model: str) -> Callable[[str, str], str]:
    def chat(system: str, user: str) -> str:
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
                "generationConfig": {"temperature": 0},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    return chat


def _ollama(model: str) -> Callable[[str, str], str]:
    def chat(system: str, user: str) -> str:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    return chat


def _mock(system: str, user: str) -> str:
    """Deterministic fake model: only knows the harness it was handed."""
    del user
    if "HARNESS: planner" in system:
        return "PLANNER"
    if "HARNESS: writer" in system:
        return "WRITER"
    return "UNKNOWN"


def resolve_provider(name: str) -> tuple[str, Callable[[str, str], str]]:
    openai = os.environ.get("OPENAI_API_KEY")
    deepseek = os.environ.get("DEEPSEEK_API_KEY")
    gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if name == "auto":
        if openai:
            name = "openai"
        elif deepseek:
            name = "deepseek"
        elif gemini:
            name = "gemini"
        else:
            name = "mock"

    if name == "openai":
        if not openai:
            raise SystemExit("OPENAI_API_KEY is not set")
        return name, _chat_completions("https://api.openai.com/v1", openai, os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    if name == "deepseek":
        if not deepseek:
            raise SystemExit("DEEPSEEK_API_KEY is not set")
        return name, _chat_completions("https://api.deepseek.com/v1", deepseek, os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    if name == "gemini":
        if not gemini:
            raise SystemExit("GEMINI_API_KEY is not set")
        return name, _gemini(gemini, os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    if name == "ollama":
        return name, _ollama(os.environ.get("OLLAMA_MODEL", "llama3.2"))
    if name == "mock":
        return name, _mock
    raise SystemExit(f"unknown provider: {name}")


class State(HarnessState):
    messages: list
    outputs: dict[str, str]


def build_graph(chat: Callable[[str, str], str]):
    @harnessed(PLANNER, validate=True)
    def planner(state: State):
        answer = chat(harness_prompt(), QUESTION)
        return {
            "messages": [answer],
            "outputs": {**state.get("outputs", {}), "planner": answer, "planner_harness": current_harness().name},
        }

    @harnessed(WRITER, validate=True)
    def writer(state: State):
        answer = chat(harness_prompt(), QUESTION)
        return {
            "messages": [answer],
            "outputs": {**state.get("outputs", {}), "writer": answer, "writer_harness": current_harness().name},
        }

    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("writer", writer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "writer")
    builder.add_edge("writer", END)
    return builder.compile()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="auto", choices=["auto", "mock", "openai", "deepseek", "gemini", "ollama"])
    parser.add_argument("--mock", action="store_true", help="shortcut for --provider mock")
    args = parser.parse_args()

    provider, chat = resolve_provider("mock" if args.mock else args.provider)
    print(f"provider: {provider}")

    out = build_graph(chat).invoke({"messages": [], "outputs": {}})
    outputs = out["outputs"]
    print("planner output:", outputs["planner"])
    print("writer output :", outputs["writer"])

    planner_ok = "PLANNER" in outputs["planner"].upper()
    writer_ok = "WRITER" in outputs["writer"].upper()
    print("planner followed its own harness:", planner_ok)
    print("writer followed its own harness :", writer_ok)

    if not (planner_ok and writer_ok):
        print("FAILED: at least one node did not follow its own harness")
        return 1

    print("OK: each node parsed and followed its own harness")
    return 0


if __name__ == "__main__":
    sys.exit(main())
