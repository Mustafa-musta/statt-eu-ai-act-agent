"""LangGraph ReAct-style agent over GPT-4o-mini.

The agent is deliberately small: a single LLM with a tools list and a
system prompt. The LangGraph ``create_react_agent`` helper handles the
loop — tool call, tool result, next decision — until the model emits a
final answer.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.tools import get_tools

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a public-policy research assistant specialised in the EU AI Act \
(Regulation (EU) 2024/1689).

You have these tools:

1. search_policy_docs(query) — semantic search over a curated corpus of EU AI Act \
documents covering: overview, risk classification, prohibited practices, high-risk \
systems, GPAI obligations, penalties, and the implementation timeline. ALWAYS try \
this first for any EU AI Act question.

2. lookup_document(filename) — fetch a full document from the corpus when a \
search snippet is too narrow. Available filenames are returned by the tool itself \
if you call it with an unknown name.

3. web_search(query) — live web search. Use ONLY as a fallback: when the user \
asks about something outside the EU AI Act, or about recent events that the \
corpus does not cover (e.g. enforcement actions, new Commission guidance, \
specific company filings).

Answering rules:
- Ground every factual claim in retrieved content. If you cannot, say so plainly.
- Cite sources inline. Use [Source: <filename>] for corpus passages and \
[Web: <url>] for web results.
- Prefer the corpus. Do not call web_search if the corpus already answers the question.
- Be concise. Two to six sentences for most questions; a short bullet list when \
the answer is genuinely enumerative (e.g. listing prohibited practices).
- This is a research tool, not legal advice. If asked for legal advice, give the \
factual content and add one sentence noting the limitation.
"""


def build_agent(model: str = DEFAULT_MODEL, temperature: float = 0.0, gov_only: bool = False):
    """Build and return a LangGraph ReAct agent.

    The returned object is invoked as::

        result = agent.invoke({"messages": [("user", "your question")]})
        answer = result["messages"][-1].content

    Pass gov_only=True to restrict web search to official EU sources.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    return create_react_agent(llm, get_tools(gov_only=gov_only), prompt=SYSTEM_PROMPT)
