"""Command-line interface — minimal REPL over the agent.

Usage:
    python cli.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from agent.agent import build_agent
from agent.rag import build_index, index_exists


def _ensure_index() -> None:
    if not index_exists():
        print("Building vector index (one-time setup)...")
        build_index()
        print("Index built.\n")


def main() -> None:
    load_dotenv()
    _ensure_index()
    agent = build_agent()
    print("EU AI Act Policy Agent. Type 'exit' or Ctrl-D to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            return

        result = agent.invoke({"messages": [("user", q)]})
        final = result["messages"][-1]

        # Show which tools the agent used.
        used: list[str] = []
        for msg in result["messages"]:
            for tc in getattr(msg, "tool_calls", []) or []:
                used.append(tc["name"])
        if used:
            print(f"[tools used: {', '.join(used)}]")

        print(f"\nAgent: {final.content}\n")


if __name__ == "__main__":
    sys.exit(main())
