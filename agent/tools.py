"""Three tools exposed to the agent.

The agent decides per-query which combination of tools to call. This is
the demonstration of *agentic* behaviour: a vanilla RAG pipeline always
retrieves; this agent retrieves only when the corpus is the right place
to look, and falls back to the live web for anything outside it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.tools import tool

from agent.rag import DEFAULT_DOCS_DIR, get_vectorstore

# ---------------------------------------------------------------------------
# Tool 1 — semantic search over the local corpus
# ---------------------------------------------------------------------------


@tool
def search_policy_docs(query: str) -> str:
    """Search the local EU AI Act document corpus for passages relevant to
    the query. Returns up to four passages, each prefixed with its source
    filename. Use this FIRST for any question about the EU AI Act."""
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=4)
    if not results:
        return "No relevant passages found in the local corpus."

    blocks: List[str] = []
    for d in results:
        fname = d.metadata.get("filename") or Path(
            d.metadata.get("source", "unknown")
        ).name
        blocks.append(f"[Source: {fname}]\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Tool 2 — direct document fetch
# ---------------------------------------------------------------------------


@tool
def lookup_document(filename: str) -> str:
    """Fetch the full plain-text content of a single document in the local
    corpus by filename (e.g. '06_penalties_enforcement.md'). Use when a
    semantic-search snippet is too narrow and you need broader context
    from one document."""
    safe = Path(filename).name  # strip any path components
    target = DEFAULT_DOCS_DIR / safe
    if not target.exists():
        available = sorted(p.name for p in DEFAULT_DOCS_DIR.glob("*.md"))
        return (
            f"Document not found: {safe}. "
            f"Available files: {', '.join(available)}"
        )
    return target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tool 3 — live web search (Tavily)
# ---------------------------------------------------------------------------

# Lazy import so the rest of the system still works if Tavily isn't installed
# or the key isn't set.
try:
    from langchain_tavily import TavilySearch as TavilySearchResults

    _tavily = TavilySearchResults(
        max_results=3,
        name="web_search",
        description=(
            "Search the live web for current information. Use ONLY when the "
            "local corpus does not contain the answer (e.g., very recent news, "
            "enforcement actions after the corpus was written, or topics "
            "outside the EU AI Act). Each result includes a URL — cite it."
        ),
    )
    _HAS_TAVILY = bool(os.getenv("TAVILY_API_KEY"))
except Exception:  # pragma: no cover
    _tavily = None
    _HAS_TAVILY = False


def get_tools() -> list:
    """Return the active tool list. Web search is included only when a
    Tavily API key is configured."""
    tools = [search_policy_docs, lookup_document]
    if _HAS_TAVILY and _tavily is not None:
        tools.append(_tavily)
    return tools
