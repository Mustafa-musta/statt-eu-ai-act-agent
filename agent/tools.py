"""Tools exposed to the agent.

make_search_tool() builds a document-scoped search tool dynamically bound
to whichever FAISS index the user loaded for their selected document.
"""

from __future__ import annotations

from typing import List
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_community.vectorstores import FAISS


def make_search_tool(vs: FAISS) -> StructuredTool:
    """Return a search tool scoped to the provided in-memory FAISS index."""

    def _search(query: str) -> str:
        results = vs.similarity_search(query, k=5)
        if not results:
            return "No relevant passages found in the loaded document."
        blocks: List[str] = []
        for d in results:
            title = d.metadata.get("title", "Document")
            blocks.append(f"[Source: {title}]\n{d.page_content.strip()}")
        return "\n\n---\n\n".join(blocks)

    return StructuredTool.from_function(
        func=_search,
        name="search_document",
        description=(
            "Search the currently loaded document for passages relevant to the query. "
            "ALWAYS call this before answering any question about the document."
        ),
    )
