"""Streamlit chat UI for the EU AI Act Policy Agent.

Run locally:
    streamlit run app.py

On Streamlit Community Cloud, set OPENAI_API_KEY and TAVILY_API_KEY in
the app's Secrets panel (TOML format). The vector index is built lazily
on the first run if ``chroma_db/`` is not yet present.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent.agent import build_agent
from agent.rag import build_index, index_exists

# ---------------------------------------------------------------------------
# Secrets / env wiring
# ---------------------------------------------------------------------------

load_dotenv()


def _hydrate_secrets() -> None:
    """Promote Streamlit Cloud secrets into environment variables.

    Accessing ``st.secrets`` can raise ``StreamlitSecretNotFoundError``
    when no ``.streamlit/secrets.toml`` exists (typical local-dev case).
    Treat that as "no Cloud secrets" and rely on .env / OS env.
    """
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        try:
            value = secrets[key]
        except Exception:
            continue
        if value:
            os.environ[key] = str(value)


_hydrate_secrets()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EU AI Act Policy Agent",
    page_icon=":scroll:",
    layout="centered",
)

st.title("EU AI Act Policy Agent")
st.caption(
    "Agentic RAG over a curated corpus of EU AI Act primary content, "
    "with a live web-search fallback."
)


# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _bootstrap_index_and_agent():
    if not index_exists():
        with st.spinner("Building the vector index for the first time..."):
            build_index()
    return build_agent()


if "OPENAI_API_KEY" not in os.environ:
    st.error(
        "No `OPENAI_API_KEY` set. Add it to your `.env` file (locally) or to the "
        "Streamlit Cloud Secrets panel before continuing."
    )
    st.stop()

agent = _bootstrap_index_and_agent()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "This agent answers questions about the **EU AI Act** "
        "(Regulation (EU) 2024/1689). It uses three tools and decides "
        "per-query which to invoke:\n\n"
        "1. `search_policy_docs` — semantic search over the local corpus.\n"
        "2. `lookup_document` — fetch a full document by filename.\n"
        "3. `web_search` — Tavily, used only when the corpus is insufficient."
    )
    st.markdown("---")
    st.subheader("Try")
    examples = [
        "What practices are prohibited under Article 5?",
        "When do GPAI obligations apply?",
        "What is the maximum fine for breaching Article 5?",
        "How does the Act define a systemic-risk GPAI model?",
        "Which use cases count as high-risk under Annex III?",
        "Have there been any recent EU AI Act enforcement actions?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_input"] = ex
    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("tool_trace"):
            with st.expander("Tool calls", expanded=False):
                for line in m["tool_trace"]:
                    st.markdown(f"- {line}")
        st.markdown(m["content"])


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask a question about the EU AI Act...")
if not prompt and "pending_input" in st.session_state:
    prompt = st.session_state.pop("pending_input")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": [("user", prompt)]})
            final = result["messages"][-1]

            # Collect tool-call trace.
            trace: list[str] = []
            for msg in result["messages"]:
                for tc in getattr(msg, "tool_calls", []) or []:
                    args_preview = ", ".join(
                        f"{k}={v!r}" for k, v in (tc.get("args") or {}).items()
                    )
                    trace.append(f"**{tc['name']}**({args_preview})")

        if trace:
            with st.expander("Tool calls", expanded=False):
                for line in trace:
                    st.markdown(f"- {line}")
        st.markdown(final.content)

    st.session_state.messages.append(
        {"role": "assistant", "content": final.content, "tool_trace": trace}
    )

if __name__ == "__main__":
    import subprocess
    import sys

    subprocess.run(["streamlit", "run", __file__] + sys.argv[1:])
