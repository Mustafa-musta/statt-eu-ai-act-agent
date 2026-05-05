"""LangGraph ReAct agent for document Q&A."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.tools import make_search_tool

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a policy research assistant. The user has loaded a specific document \
for analysis. Your job is to answer their questions using that document's content.

You have one tool:
  search_document(query) — semantic search over the loaded document.

Rules:
- ALWAYS call search_document before answering. Never answer from memory alone.
- Cite every factual claim inline as [Source: <document title>].
- If the document does not cover something, say so clearly rather than guessing.
- Be concise: 2–5 sentences for most answers; bullet lists for enumerable facts.
- This is a research tool. If asked for professional advice (legal, medical, financial), \
give the factual content from the document and note the limitation in one sentence.
"""


def build_doc_agent(vs: FAISS, model: str = DEFAULT_MODEL):
    """Build a ReAct agent scoped to the given in-memory FAISS index."""
    llm = ChatOpenAI(model=model, temperature=0.0)
    return create_react_agent(llm, [make_search_tool(vs)], prompt=SYSTEM_PROMPT)
