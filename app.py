"""Policy Research Agent — multi-doc, multi-source, with preview."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Secrets
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
# Constants
# ---------------------------------------------------------------------------

MAX_DOCS = 6

TOPICS: dict[str, dict] = {
    "Climate Change": {
        "icon": "🌍",
        "color": "#16a34a",
        "gov_domains": ["unfccc.int", "epa.gov", "ec.europa.eu", "ipcc.ch", "iea.org", "climate.gov", "unep.org"],
        "default_query": "climate change policy report 2024",
    },
    "Healthcare Reform": {
        "icon": "🏥",
        "color": "#dc2626",
        "gov_domains": ["who.int", "nih.gov", "hhs.gov", "cms.gov", "ec.europa.eu", "ncbi.nlm.nih.gov", "paho.org"],
        "default_query": "healthcare reform policy analysis 2024",
    },
    "Education Policy": {
        "icon": "📚",
        "color": "#2563eb",
        "gov_domains": ["ed.gov", "oecd.org", "unesco.org", "ec.europa.eu", "nces.ed.gov", "uis.unesco.org"],
        "default_query": "education policy reform report 2024",
    },
}

SOURCE_META: dict[str, dict] = {
    "Government & Official": {"icon": "🏛️", "domains_key": "gov_domains"},
    "News & Media":          {"icon": "📰", "domains_key": None},
    "Wikipedia":             {"icon": "📖", "domains_key": None, "fixed_domains": ["en.wikipedia.org"]},
}

SOURCE_BADGE_COLOR = {
    "Government & Official": "#1d4ed8",
    "News & Media":          "#b45309",
    "Wikipedia":             "#6d28d9",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px !important; border: 1px solid #334155 !important;
    background-color: #1e293b !important; color: #e2e8f0 !important;
    font-size: 13px !important; text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background-color: #334155 !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #4f46e5 !important; border-color: #4f46e5 !important; font-weight: 700 !important;
}
.doc-card {
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 14px; margin-bottom: 8px; background: #f8fafc;
}
.doc-title { font-weight: 700; font-size: 14px; margin-bottom: 3px; }
.doc-url   { font-size: 11px; color: #64748b; margin-bottom: 5px; word-break: break-all; }
.doc-snippet { font-size: 12px; color: #374151; }
.src-badge {
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 600; color: white; margin-left: 6px;
}
.loaded-doc-row {
    display: flex; align-items: center; gap: 8px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 7px 10px; margin-bottom: 5px; font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_defaults: dict = {
    "active_topic":   "Climate Change",
    "active_sources": ["Government & Official"],  # list — multi-source
    "search_results": [],
    "loaded_docs":    [],          # [{"title","url","source"}, ...]  max 6
    "doc_texts":      {},          # url -> (text, title)  for rebuild
    "vs":             None,        # combined FAISS index
    "agent":          None,
    "messages":       [],
    "previews":       {},          # url -> text (cached preview content)
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domains_for(source: str, topic: str) -> list[str] | None:
    meta = SOURCE_META[source]
    if "fixed_domains" in meta:
        return meta["fixed_domains"]
    key = meta["domains_key"]
    if key:
        return TOPICS[topic][key]
    return None


def _scoped_query(query: str, topic: str) -> str:
    """Ensure the topic name is part of the search query."""
    if topic.lower() in query.lower():
        return query
    return f"{topic} {query}"


def _multi_source_search(query: str, topic: str, sources: list[str]) -> list[dict]:
    from tavily import TavilyClient
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return []
    client = TavilyClient(api_key=key)
    scoped = _scoped_query(query, topic)
    per_src = max(2, MAX_DOCS // len(sources))
    seen: set[str] = set()
    results: list[dict] = []
    for src in sources:
        kwargs: dict = {"query": scoped, "max_results": per_src + 1}
        domains = _domains_for(src, topic)
        if domains:
            kwargs["include_domains"] = domains
        try:
            for r in client.search(**kwargs).get("results", []):
                url = r.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    r["_source"] = src
                    results.append(r)
        except Exception:
            pass
    return results[:MAX_DOCS]


def _is_pdf(url: str, content_type: str = "") -> bool:
    return url.lower().endswith(".pdf") or "application/pdf" in content_type


def _extract_pdf(data: bytes) -> str:
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception:
        return ""


def _fetch_text(url: str) -> str:
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}

    # --- PDF path ---
    if _is_pdf(url):
        try:
            r = requests.get(url, timeout=30, headers=headers)
            r.raise_for_status()
            return _extract_pdf(r.content)
        except Exception:
            return ""

    # --- HTML path: try trafilatura first ---
    try:
        import trafilatura
        dl = trafilatura.fetch_url(url)
        if dl:
            # trafilatura may have fetched a PDF redirect
            if _is_pdf(url) or b"%PDF" in dl[:8]:
                return _extract_pdf(dl if isinstance(dl, bytes) else dl.encode())
            text = trafilatura.extract(dl, include_tables=False)
            if text and len(text) > 200:
                return text
    except Exception:
        pass

    # --- HTML fallback: requests + simple stripper ---
    try:
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if _is_pdf(url, ct):
            return _extract_pdf(r.content)
        from html.parser import HTMLParser
        class _P(HTMLParser):
            def __init__(self):
                super().__init__(); self.out = []; self._skip = False
            def handle_starttag(self, t, _):
                if t in ("script", "style"): self._skip = True
            def handle_endtag(self, t):
                if t in ("script", "style"): self._skip = False
            def handle_data(self, d):
                if not self._skip: self.out.append(d)
        p = _P(); p.feed(r.text)
        return " ".join(p.out)
    except Exception:
        return ""


def _add_document(text: str, title: str, url: str, source: str) -> None:
    """Embed a document and merge it into the combined FAISS index."""
    from agent.rag import build_doc_index
    from agent.agent import build_doc_agent
    new_vs = build_doc_index(text, title=title, source_url=url)
    st.session_state.doc_texts[url] = (text, title)
    if st.session_state.vs is None:
        st.session_state.vs = new_vs
    else:
        st.session_state.vs.merge_from(new_vs)
    st.session_state.loaded_docs.append({"title": title, "url": url, "source": source})
    st.session_state.agent = build_doc_agent(st.session_state.vs, topic=st.session_state.active_topic)


def _remove_document(url: str) -> None:
    """Remove a document and rebuild the combined index from remaining texts."""
    from agent.rag import build_doc_index
    from agent.agent import build_doc_agent
    st.session_state.doc_texts.pop(url, None)
    st.session_state.loaded_docs = [d for d in st.session_state.loaded_docs if d["url"] != url]
    st.session_state.messages = []
    if not st.session_state.doc_texts:
        st.session_state.vs = None
        st.session_state.agent = None
        return
    items = list(st.session_state.doc_texts.items())
    u0, (t0, ti0) = items[0]
    vs = build_doc_index(t0, title=ti0, source_url=u0)
    for u, (t, ti) in items[1:]:
        vs.merge_from(build_doc_index(t, title=ti, source_url=u))
    st.session_state.vs = vs
    st.session_state.agent = build_doc_agent(vs, topic=st.session_state.active_topic)

# ---------------------------------------------------------------------------
# API key guard
# ---------------------------------------------------------------------------

if "OPENAI_API_KEY" not in os.environ:
    st.error("No `OPENAI_API_KEY` found. Add it to `.env` or Streamlit Cloud Secrets.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔍 Policy Research Agent")
    st.caption("Multi-doc · Multi-source · RAG Q&A")
    st.divider()

    st.markdown("### 📂 Topic")
    for topic, meta in TOPICS.items():
        is_active = st.session_state.active_topic == topic
        if st.button(f"{meta['icon']}  {topic}", key=f"t_{topic}",
                     use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.active_topic = topic
            st.session_state.search_results = []
            st.rerun()

    st.divider()
    st.markdown("### 🔎 Sources  *(select multiple)*")
    for src in SOURCE_META:
        checked = src in st.session_state.active_sources
        icon = SOURCE_META[src]["icon"]
        if st.checkbox(f"{icon}  {src}", value=checked, key=f"src_{src}"):
            if src not in st.session_state.active_sources:
                st.session_state.active_sources.append(src)
                st.session_state.search_results = []
                st.rerun()
        else:
            if src in st.session_state.active_sources and len(st.session_state.active_sources) > 1:
                st.session_state.active_sources.remove(src)
                st.session_state.search_results = []
                st.rerun()

    st.divider()
    loaded = st.session_state.loaded_docs
    if loaded:
        st.markdown(f"### 📄 Loaded Documents ({len(loaded)}/{MAX_DOCS})")
        for d in loaded:
            st.caption(f"• {d['title'][:45]}{'…' if len(d['title']) > 45 else ''}")
        if st.button("🗑️ Clear all documents", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()
    else:
        if st.button("🗑️ Clear session", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

topic_meta = TOPICS[st.session_state.active_topic]
color = topic_meta["color"]
src_labels = " · ".join(
    f"{SOURCE_META[s]['icon']} {s}" for s in st.session_state.active_sources
)

st.markdown(
    f"<h2 style='color:{color}'>{topic_meta['icon']} {st.session_state.active_topic} — Policy Research</h2>",
    unsafe_allow_html=True,
)
st.caption(f"Sources: {src_labels}")

col_docs, col_chat = st.columns([1, 1], gap="large")

# ---------------------------------------------------------------------------
# LEFT — Document discovery
# ---------------------------------------------------------------------------

with col_docs:
    st.markdown(
        f"### 📑 Find Documents — "
        f"<span style='color:{color}'>{topic_meta['icon']} {st.session_state.active_topic}</span>",
        unsafe_allow_html=True,
    )
    st.caption("Search is scoped to the selected topic. Switch topics in the sidebar.")
    remaining = MAX_DOCS - len(st.session_state.loaded_docs)
    if remaining == 0:
        st.info(f"Maximum of {MAX_DOCS} documents loaded. Remove one to add more.")

    query = st.text_input(
        "query", value=topic_meta["default_query"],
        placeholder="Enter a search query...", label_visibility="collapsed",
    )

    if st.button("🔍 Find Documents", use_container_width=True, type="primary", disabled=(remaining == 0)):
        if not os.environ.get("TAVILY_API_KEY"):
            st.warning("Set `TAVILY_API_KEY` in Secrets to enable document search.")
        else:
            with st.spinner("Searching across selected sources..."):
                results = _multi_source_search(
                    query, st.session_state.active_topic, st.session_state.active_sources
                )
            st.session_state.search_results = results
            if not results:
                st.info("No results found. Try a different query or source.")

    # Result cards
    loaded_urls = {d["url"] for d in st.session_state.loaded_docs}
    results = st.session_state.search_results

    if results:
        st.markdown(f"**{len(results)} documents found:**")
        for i, r in enumerate(results):
            title   = r.get("title") or f"Document {i+1}"
            url     = r.get("url", "")
            snippet = (r.get("content") or r.get("snippet") or "")[:200]
            src     = r.get("_source", "")
            badge_color = SOURCE_BADGE_COLOR.get(src, "#475569")

            already_loaded = url in loaded_urls
            src_badge = f"<span class='src-badge' style='background:{badge_color}'>{SOURCE_META.get(src,{}).get('icon','')} {src}</span>"

            st.markdown(
                f"<div class='doc-card'>"
                f"<div class='doc-title'>{title}{src_badge}</div>"
                f"<div class='doc-url'>{url}</div>"
                f"<div class='doc-snippet'>{snippet}{'…' if len(snippet)==200 else ''}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            btn_col, prev_col = st.columns([1, 1])

            # Preview button
            with prev_col:
                prev_key = f"show_prev_{i}"
                if prev_key not in st.session_state:
                    st.session_state[prev_key] = False
                if st.button("👁️ Preview", key=f"prev_btn_{i}", use_container_width=True):
                    if url not in st.session_state.previews:
                        with st.spinner("Fetching preview..."):
                            text = _fetch_text(url)
                            st.session_state.previews[url] = text
                    st.session_state[prev_key] = not st.session_state[prev_key]
                    st.rerun()

            # Load button
            with btn_col:
                if already_loaded:
                    st.button("✅ Loaded", key=f"load_{i}", disabled=True, use_container_width=True)
                elif remaining == 0:
                    st.button("📥 Add", key=f"load_{i}", disabled=True, use_container_width=True)
                else:
                    if st.button("📥 Add to Analysis", key=f"load_{i}", use_container_width=True):
                        with st.spinner(f"Indexing '{title[:40]}'..."):
                            text = st.session_state.previews.get(url) or _fetch_text(url)
                            if not text or len(text) < 100:
                                st.error("Could not extract enough text. Try another document.")
                            else:
                                _add_document(text, title=title, url=url, source=src)
                                loaded_urls.add(url)
                                remaining -= 1
                                st.rerun()

            # Preview panel
            if st.session_state.get(f"show_prev_{i}"):
                preview_text = st.session_state.previews.get(url, "")
                with st.expander("📄 Document Preview", expanded=True):
                    if not preview_text:
                        st.warning("Could not fetch content for this URL.")
                    else:
                        st.text_area(
                            "preview",
                            value=preview_text[:2000] + ("\n\n[truncated…]" if len(preview_text) > 2000 else ""),
                            height=250,
                            label_visibility="collapsed",
                            disabled=True,
                        )

# ---------------------------------------------------------------------------
# RIGHT — Q&A
# ---------------------------------------------------------------------------

with col_chat:
    st.markdown("### 💬 Ask Questions")

    loaded = st.session_state.loaded_docs
    if not loaded:
        st.info("👈 Search for documents, preview them, then click **Add to Analysis** to start asking questions.")
    else:
        # Loaded documents list with remove buttons
        st.markdown(f"**{len(loaded)}/{MAX_DOCS} documents loaded:**")
        for d in loaded:
            badge_color = SOURCE_BADGE_COLOR.get(d.get("source", ""), "#475569")
            c1, c2 = st.columns([5, 1])
            with c1:
                src_badge = f"<span class='src-badge' style='background:{badge_color}'>{SOURCE_META.get(d.get('source',''),{}).get('icon','')} {d.get('source','')}</span>"
                st.markdown(
                    f"<div style='font-size:13px;padding:4px 0'>"
                    f"📄 <strong>{d['title'][:55]}{'…' if len(d['title'])>55 else ''}</strong>"
                    f"{src_badge}</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"rm_{d['url']}", help="Remove this document"):
                    with st.spinner("Rebuilding index..."):
                        _remove_document(d["url"])
                    st.rerun()

        st.divider()

        # Chat history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                if m.get("tool_trace"):
                    with st.expander("🔧 Tool calls", expanded=False):
                        for line in m["tool_trace"]:
                            st.markdown(f"- {line}")
                st.markdown(m["content"])

        # Chat input
        n_docs = len(loaded)
        prompt = st.chat_input(
            f"Ask a question across {n_docs} loaded document{'s' if n_docs > 1 else ''}..."
        )
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke({"messages": [("user", prompt)]})
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
