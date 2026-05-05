"""Streamlit UI for the EU AI Act Policy Agent."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent.agent import build_agent
from agent.rag import build_index, index_exists

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
    page_title="EU AI Act Policy Agent",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Topic definitions (map to the 7 corpus documents)
# ---------------------------------------------------------------------------

TOPICS: dict[str, dict] = {
    "All Topics": {
        "icon": "🌐",
        "color": "#6366f1",
        "file": None,
        "description": "Search across the full EU AI Act corpus.",
        "queries": [
            "What is the EU AI Act?",
            "Who does the EU AI Act apply to?",
            "What are the key obligations for AI providers?",
        ],
    },
    "Overview": {
        "icon": "📋",
        "color": "#3b82f6",
        "file": "01_overview.md",
        "description": "Scope, objectives, and general structure of the regulation.",
        "queries": [
            "What is the scope of the EU AI Act?",
            "Which organisations does the Act apply to?",
            "What are the main obligations under the Act?",
        ],
    },
    "Risk Classification": {
        "icon": "⚖️",
        "color": "#f59e0b",
        "file": "02_risk_classification.md",
        "description": "Four-tier risk pyramid: unacceptable, high, limited, and minimal.",
        "queries": [
            "How does the Act classify AI risk levels?",
            "What is a minimal-risk AI system?",
            "What is the difference between limited and high risk?",
        ],
    },
    "Prohibited Practices": {
        "icon": "🚫",
        "color": "#ef4444",
        "file": "03_prohibited_practices.md",
        "description": "AI uses banned outright under Article 5.",
        "queries": [
            "What practices are prohibited under Article 5?",
            "Is social scoring allowed under the EU AI Act?",
            "Are subliminal manipulation techniques banned?",
        ],
    },
    "High-Risk Systems": {
        "icon": "🔴",
        "color": "#dc2626",
        "file": "04_high_risk_systems.md",
        "description": "Annex III categories, conformity assessments, and provider duties.",
        "queries": [
            "Which use cases count as high-risk under Annex III?",
            "What are the obligations for high-risk AI providers?",
            "Does biometric identification count as high-risk?",
        ],
    },
    "GPAI Obligations": {
        "icon": "🤖",
        "color": "#10b981",
        "file": "05_gpai_obligations.md",
        "description": "General-purpose AI models, systemic-risk thresholds, and transparency duties.",
        "queries": [
            "When do GPAI obligations apply?",
            "How does the Act define a systemic-risk GPAI model?",
            "What is the 10²⁵ FLOP threshold for?",
        ],
    },
    "Penalties": {
        "icon": "💰",
        "color": "#8b5cf6",
        "file": "06_penalties_enforcement.md",
        "description": "Fine tiers, enforcement bodies, and market surveillance.",
        "queries": [
            "What is the maximum fine for breaching Article 5?",
            "What are the penalties for GPAI providers?",
            "How are fines calculated under the Act?",
        ],
    },
    "Timeline": {
        "icon": "📅",
        "color": "#06b6d4",
        "file": "07_implementation_timeline.md",
        "description": "Phase-in dates, transitional periods, and key deadlines.",
        "queries": [
            "When did the EU AI Act enter into force?",
            "What is the implementation timeline for high-risk systems?",
            "When do GPAI rules take effect?",
        ],
    },
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    text-align: left !important;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #334155 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #4f46e5 !important;
    border-color: #4f46e5 !important;
    font-weight: 700 !important;
}
.topic-header {
    padding: 12px 18px;
    border-radius: 10px;
    margin-bottom: 8px;
    color: white !important;
    font-size: 22px;
    font-weight: 700;
}
.suggestion-label {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_topic" not in st.session_state:
    st.session_state.active_topic = "All Topics"
if "gov_only" not in st.session_state:
    st.session_state.gov_only = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚖️ EU AI Act Agent")
    st.caption("Agentic RAG · GPT-4o-mini · FAISS")
    st.divider()

    st.markdown("### 📂 Topics")
    for topic, meta in TOPICS.items():
        is_active = st.session_state.active_topic == topic
        label = f"{meta['icon']}  {topic}"
        if st.button(label, key=f"topic_{topic}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_topic = topic
            st.rerun()

    st.divider()
    st.markdown("### 🌐 Web Search Source")
    search_mode = st.radio(
        "source",
        options=["All news & sources", "Government & official only"],
        index=1 if st.session_state.gov_only else 0,
        label_visibility="collapsed",
    )
    new_gov_only = (search_mode == "Government & official only")
    if new_gov_only != st.session_state.gov_only:
        st.session_state.gov_only = new_gov_only
        st.rerun()

    if st.session_state.gov_only:
        st.caption(
            "🏛️ Restricted to official EU sources:\n"
            "ec.europa.eu · eur-lex.europa.eu · europarl.europa.eu · "
            "consilium.europa.eu · enisa.europa.eu"
        )
    else:
        st.caption("🔓 All public sources including news outlets and research.")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Bootstrap agent (cached per gov_only setting)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _bootstrap(gov_only: bool):
    if not index_exists():
        with st.spinner("Building vector index for the first time — please wait..."):
            build_index()
    return build_agent(gov_only=gov_only)

if "OPENAI_API_KEY" not in os.environ:
    st.error(
        "No `OPENAI_API_KEY` found. Add it to your `.env` file or "
        "Streamlit Cloud **Secrets** panel."
    )
    st.stop()

agent = _bootstrap(st.session_state.gov_only)

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

active = TOPICS[st.session_state.active_topic]
color = active["color"]

st.markdown(
    f"<div class='topic-header' style='background:{color}'>"
    f"{active['icon']} EU AI Act — {st.session_state.active_topic}"
    f"</div>",
    unsafe_allow_html=True,
)
st.caption(active["description"])

# Search source badge
mode_label = "🏛️ Government & official sources only" if st.session_state.gov_only else "🔓 All news & sources"
st.markdown(f"**Web search:** {mode_label}")

st.divider()

# ---------------------------------------------------------------------------
# Suggested queries
# ---------------------------------------------------------------------------

st.markdown("<div class='suggestion-label'>💡 Suggested questions — click to ask:</div>", unsafe_allow_html=True)
cols = st.columns(3)
for i, q in enumerate(active["queries"]):
    if cols[i].button(q, use_container_width=True, key=f"sugg_{i}"):
        st.session_state["pending_input"] = q

st.divider()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("tool_trace"):
            with st.expander("🔧 Tool calls", expanded=False):
                for line in m["tool_trace"]:
                    st.markdown(f"- {line}")
        st.markdown(m["content"])

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

placeholder = (
    f"Ask about {st.session_state.active_topic}..."
    if st.session_state.active_topic != "All Topics"
    else "Ask a question about the EU AI Act..."
)
prompt = st.chat_input(placeholder)
if not prompt and "pending_input" in st.session_state:
    prompt = st.session_state.pop("pending_input")

if prompt:
    # Inject topic context into the query when a specific topic is active
    scoped = prompt
    if st.session_state.active_topic != "All Topics" and active["file"]:
        scoped = (
            f"[Topic focus: {st.session_state.active_topic} "
            f"(document: {active['file']})] {prompt}"
        )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": [("user", scoped)]})
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
