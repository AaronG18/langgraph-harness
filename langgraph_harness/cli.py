"""Command line interface: validate text against a harness without writing Python.

Two uses:

* in CI, gate a fixture or a golden file against the harness that produced it
* while debugging, print exactly what a node would receive as its system prompt

Exit codes follow the usual convention so a shell can branch on them:
``0`` pass, ``1`` violation, ``2`` usage or parse error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import HarnessError
from .harness import Harness
from .validation import Report, validate_output

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_ERROR = 2

# A starter document that already uses sections, so `init` -> `check` works in one step.
STARTER_TEMPLATE = """# {name} harness

Replace these examples with your own rules. `MUST` / `MUST NOT` are enforced,
`SHOULD` is reported as a warning, and each `##` heading groups the rules below it
so a gate can act on one group.

## Output format
- MUST: return a JSON object with a "steps" key [require=\\{{\\s*"steps"]

## Tone
- MUST NOT: mention internal tool names [forbid=tool_]
- SHOULD: keep the answer under 200 words
"""


def _read_text(source: str | None) -> str:
    """Read the candidate output from a file, ``-``/nothing meaning stdin."""
    if source is None or source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")


def _emit(report: Report, *, as_json: bool, strict: bool, name: str) -> int:
    warnings_fail = strict and bool(report.warnings)
    failed = not report.ok or warnings_fail

    if as_json:
        payload = {
            "harness": name,
            "ok": not failed,
            "violations": [v.as_dict() for v in report.violations],
            "warnings": [v.as_dict() for v in report.warnings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for violation in report.violations:
            print(f"VIOLATION {violation.rule_id} [{violation.level}] {violation.message}")
        for warning in report.warnings:
            prefix = "VIOLATION" if strict else "WARNING"
            print(f"{prefix} {warning.rule_id} [{warning.level}] {warning.message}")
        if not failed:
            print(f"OK: {name} satisfied")

    return EXIT_VIOLATION if failed else EXIT_OK


def _cmd_check(args: argparse.Namespace) -> int:
    harness = Harness.from_file(args.harness, name=args.name)
    if args.text_option is not None:
        text = args.text_option
    else:
        text = _read_text(args.text)
    report = validate_output(text, harness, sections=args.sections)
    return _emit(report, as_json=args.json, strict=args.strict, name=harness.name)


def _cmd_prompt(args: argparse.Namespace) -> int:
    harness = Harness.from_file(args.harness, name=args.name)
    print(harness.prompt())
    return EXIT_OK


def _cmd_diff(args: argparse.Namespace) -> int:
    old = Harness.from_file(args.old)
    new = Harness.from_file(args.new)
    result = old.diff(new)

    if args.json:
        payload = {"different": not result.is_empty, "count": len(result), **result.as_dict()}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(result.format())

    return EXIT_OK if result.is_empty else EXIT_VIOLATION


def _cmd_init(args: argparse.Namespace) -> int:
    target = args.path
    fallback = Path(target).stem if target not in ("", "-") else "plan"
    text = STARTER_TEMPLATE.format(name=args.name or fallback)

    if target == "-":
        print(text, end="")
        return EXIT_OK

    path = Path(target)
    if path.exists() and not args.force:
        print(f"error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        return EXIT_ERROR

    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langgraph-harness",
        description="Check text against a Markdown HARNESS, or print the prompt a node receives.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate text against a harness")
    check.add_argument("harness", help="path to the harness Markdown file")
    check.add_argument("text", nargs="?", help="file holding the output to check; '-' or omitted reads stdin")
    check.add_argument("--text", dest="text_option", help="validate this string instead of a file")
    check.add_argument(
        "--section",
        action="append",
        dest="sections",
        metavar="NAME",
        help="only check rules under this ## section (repeatable)",
    )
    check.add_argument("--json", action="store_true", help="emit a JSON report")
    check.add_argument("--strict", action="store_true", help="treat SHOULD warnings as failures")
    check.add_argument("--name", help="override the harness name shown in the report")
    check.set_defaults(func=_cmd_check)

    prompt = sub.add_parser("prompt", help="print the harness as a system prompt")
    prompt.add_argument("harness", help="path to the harness Markdown file")
    prompt.add_argument("--name", help="override the harness name")
    prompt.set_defaults(func=_cmd_prompt)

    diff = sub.add_parser("diff", help="compare two harness files (exit 1 when they differ)")
    diff.add_argument("old", help="the previous harness")
    diff.add_argument("new", help="the current harness")
    diff.add_argument("--json", action="store_true", help="emit a JSON report")
    diff.set_defaults(func=_cmd_diff)

    init = sub.add_parser("init", help="write a starter harness document with sections")
    init.add_argument("path", nargs="?", default="harness.md", help="where to write it; '-' prints to stdout")
    init.add_argument("--name", help="harness name used in the title (default: file stem)")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(func=_cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``langgraph-harness`` and ``python -m langgraph_harness``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (HarnessError, OSError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
