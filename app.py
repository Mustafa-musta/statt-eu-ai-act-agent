"""Policy Research Agent — find documents, load one, ask questions via RAG."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Secrets (Streamlit Cloud)
# ---------------------------------------------------------------------------

def _hydrate_secrets() -> None:
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
    page_title="Policy Research Agent",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Topic + source definitions
# ---------------------------------------------------------------------------

TOPICS: dict[str, dict] = {
    "Climate Change": {
        "icon": "🌍",
        "color": "#16a34a",
        "gov_domains": [
            "unfccc.int", "epa.gov", "ec.europa.eu", "ipcc.ch",
            "iea.org", "climate.gov", "unep.org",
        ],
        "default_query": "climate change policy report 2024",
    },
    "Healthcare Reform": {
        "icon": "🏥",
        "color": "#dc2626",
        "gov_domains": [
            "who.int", "nih.gov", "hhs.gov", "cms.gov",
            "ec.europa.eu", "ncbi.nlm.nih.gov", "paho.org",
        ],
        "default_query": "healthcare reform policy analysis 2024",
    },
    "Education Policy": {
        "icon": "📚",
        "color": "#2563eb",
        "gov_domains": [
            "ed.gov", "oecd.org", "unesco.org",
            "ec.europa.eu", "nces.ed.gov", "uis.unesco.org",
        ],
        "default_query": "education policy reform report 2024",
    },
}

SOURCES: dict[str, dict] = {
    "Government & Official": {
        "icon": "🏛️",
        "note": "Restricted to official government and intergovernmental sites.",
    },
    "News & Media": {
        "icon": "📰",
        "note": "All public news outlets, journals, and research publications.",
        "domains": None,
    },
    "Wikipedia": {
        "icon": "📖",
        "note": "Wikipedia articles only.",
        "domains": ["en.wikipedia.org"],
    },
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #334155 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #4f46e5 !important;
    border-color: #4f46e5 !important;
    font-weight: 700 !important;
}
.doc-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: #f8fafc;
}
.doc-title { font-weight: 700; font-size: 15px; margin-bottom: 4px; }
.doc-url { font-size: 11px; color: #64748b; margin-bottom: 6px; word-break: break-all; }
.doc-snippet { font-size: 13px; color: #374151; }
.loaded-banner {
    padding: 10px 16px;
    border-radius: 8px;
    background: #dcfce7;
    border-left: 4px solid #16a34a;
    font-size: 14px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = {
    "active_topic": "Climate Change",
    "active_source": "Government & Official",
    "search_results": [],
    "current_doc": None,   # {"title": str, "url": str}
    "agent": None,
    "messages": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔍 Policy Research Agent")
    st.caption("Find a document → ask questions via RAG")
    st.divider()

    st.markdown("### 📂 Topic")
    for topic, meta in TOPICS.items():
        is_active = st.session_state.active_topic == topic
        if st.button(
            f"{meta['icon']}  {topic}",
            key=f"t_{topic}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_topic = topic
            st.session_state.search_results = []
            st.rerun()

    st.divider()
    st.markdown("### 🔎 Source")
    source_choice = st.radio(
        "source",
        options=list(SOURCES.keys()),
        index=list(SOURCES.keys()).index(st.session_state.active_source),
        label_visibility="collapsed",
        format_func=lambda s: f"{SOURCES[s]['icon']}  {s}",
    )
    if source_choice != st.session_state.active_source:
        st.session_state.active_source = source_choice
        st.session_state.search_results = []
        st.rerun()
    st.caption(SOURCES[st.session_state.active_source]["note"])

    st.divider()
    if st.session_state.current_doc:
        st.markdown(f"**📄 Loaded:**")
        st.caption(st.session_state.current_doc["title"])
    if st.button("🗑️ Clear session", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

if "OPENAI_API_KEY" not in os.environ:
    st.error("No `OPENAI_API_KEY` found. Add it to `.env` or Streamlit Cloud Secrets.")
    st.stop()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tavily_search(query: str, topic: str, source: str, n: int = 6) -> list[dict]:
    """Call Tavily and return a list of result dicts."""
    from tavily import TavilyClient
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return []
    client = TavilyClient(api_key=key)
    kwargs: dict = {"query": query, "max_results": n}
    if source == "Government & Official":
        kwargs["include_domains"] = TOPICS[topic]["gov_domains"]
    elif source == "Wikipedia":
        kwargs["include_domains"] = ["en.wikipedia.org"]
    resp = client.search(**kwargs)
    return resp.get("results", [])


def _fetch_text(url: str) -> str:
    """Fetch and extract clean text from a URL using trafilatura."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_tables=False)
            if text and len(text) > 200:
                return text
    except Exception:
        pass
    # Fallback: plain requests
    try:
        import requests
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        from html.parser import HTMLParser
        class _S(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self._skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True
            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip:
                    self.parts.append(data)
        p = _S()
        p.feed(r.text)
        return " ".join(p.parts)
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Main layout — left: document discovery | right: Q&A
# ---------------------------------------------------------------------------

topic_meta = TOPICS[st.session_state.active_topic]
color = topic_meta["color"]

st.markdown(
    f"<h2 style='color:{color}'>{topic_meta['icon']} {st.session_state.active_topic} — Policy Research</h2>",
    unsafe_allow_html=True,
)

col_docs, col_chat = st.columns([1, 1], gap="large")

# ── LEFT: Document discovery ─────────────────────────────────────────────────
with col_docs:
    st.markdown("### 📑 Find Documents")

    query = st.text_input(
        "Search query",
        value=topic_meta["default_query"],
        placeholder="Enter a search query...",
        label_visibility="collapsed",
    )

    if st.button("🔍 Find Documents", use_container_width=True, type="primary"):
        if not os.environ.get("TAVILY_API_KEY"):
            st.warning("Set `TAVILY_API_KEY` in Secrets to enable document search.")
        else:
            with st.spinner("Searching..."):
                results = _tavily_search(
                    query,
                    st.session_state.active_topic,
                    st.session_state.active_source,
                )
            st.session_state.search_results = results
            if not results:
                st.info("No results found. Try a different query or source.")

    # Document result cards
    if st.session_state.search_results:
        st.markdown(f"**{len(st.session_state.search_results)} documents found** — click one to load it for Q&A:")
        for i, r in enumerate(st.session_state.search_results):
            title = r.get("title") or r.get("url", f"Document {i+1}")
            url = r.get("url", "")
            snippet = r.get("content") or r.get("snippet") or ""
            snippet = snippet[:220] + "…" if len(snippet) > 220 else snippet

            with st.container():
                st.markdown(
                    f"<div class='doc-card'>"
                    f"<div class='doc-title'>{title}</div>"
                    f"<div class='doc-url'>{url}</div>"
                    f"<div class='doc-snippet'>{snippet}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("📥 Load & Analyze", key=f"load_{i}", use_container_width=True):
                    with st.spinner(f"Fetching and indexing '{title}'..."):
                        text = _fetch_text(url)
                        if not text or len(text) < 100:
                            st.error("Could not extract enough text from this URL. Try another document.")
                        else:
                            from agent.rag import build_doc_index
                            from agent.agent import build_doc_agent
                            vs = build_doc_index(text, title=title, source_url=url)
                            st.session_state.current_doc = {"title": title, "url": url}
                            st.session_state.agent = build_doc_agent(vs)
                            st.session_state.messages = []
                            st.rerun()

# ── RIGHT: Q&A chat ──────────────────────────────────────────────────────────
with col_chat:
    st.markdown("### 💬 Ask Questions")

    if not st.session_state.current_doc:
        st.info("👈 Search for documents on the left, then click **Load & Analyze** to start asking questions.")
    else:
        doc = st.session_state.current_doc
        st.markdown(
            f"<div class='loaded-banner'>📄 <strong>{doc['title']}</strong><br>"
            f"<span style='font-size:11px;color:#166534'>{doc['url']}</span></div>",
            unsafe_allow_html=True,
        )

        # Chat history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                if m.get("tool_trace"):
                    with st.expander("🔧 Tool calls", expanded=False):
                        for line in m["tool_trace"]:
                            st.markdown(f"- {line}")
                st.markdown(m["content"])

        # Input
        prompt = st.chat_input("Ask a question about this document...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke(
                        {"messages": [("user", prompt)]}
                    )
                    final = result["messages"][-1]

                    trace: list[str] = []
                    for msg in result["messages"]:
                        for tc in getattr(msg, "tool_calls", []) or []:
                            args_preview = ", ".join(
                                f"{k}={v!r}" for k, v in (tc.get("args") or {}).items()
                            )
                            trace.append(f"**{tc['name']}**({args_preview})")

                if trace:
                    with st.expander("🔧 Tool calls", expanded=False):
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
