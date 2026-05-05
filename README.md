# Policy Research Agent

An agentic, multi-document RAG application for researching public policy topics.
Users select a topic, choose sources, discover documents from the live web, load up to six
of them into a shared vector index, and ask questions that are answered strictly from the
loaded document text.

Built for the Statt Full-Stack ML Engineer take-home.

## Topics

| Topic | Icon | Scope |
|---|---|---|
| Climate Change | 🌍 | IPCC reports, NDCs, carbon pricing, net-zero pathways |
| Healthcare Reform | 🏥 | UHC, insurance design, pharmaceutical pricing, WHO/NIH |
| Education Policy | 📚 | PISA/TALIS, curriculum, funding equity, UNESCO SDG-4 |

## Architecture

```
User selects topic + source(s)
        │
        ▼
Tavily search (topic-scoped query)
        │
        ▼
Document result cards (title · URL · snippet · source badge)
        │  👁️ Preview  │  📥 Add to Analysis
        ▼
trafilatura / pypdf content extraction
        │
        ▼
In-memory FAISS index  ──merge──▶  Combined index (up to 6 docs)
        │
        ▼
LangGraph ReAct agent (gpt-4o-mini)
  └─ search_document(query)  →  top-5 passages from combined index
        │
        ▼
Answer grounded only in retrieved passages + inline citations
```

### Agent design

The agent is a LangGraph `create_react_agent` with a **topic-specific expert persona**:

- **Climate Change** — climate policy analyst fluent in IPCC AR6, NDCs, carbon markets, 1.5 °C pathways
- **Healthcare Reform** — health systems researcher versed in UHC, DRG payment, pharmaceutical pricing
- **Education Policy** — comparative education specialist who knows PISA/TALIS, Title I, achievement gaps

The agent has one tool — `search_document(query)` — and is instructed to answer **only from retrieved passages**. If the documents do not contain the answer it responds: *"Not found in the loaded documents. I can search more documents."*

### Source modes

| Source | Restriction |
|---|---|
| 🏛️ Government & Official | Domain-restricted to topic-specific government sites (unfccc.int, who.int, ed.gov, ec.europa.eu, …) |
| 📰 News & Media | Unrestricted Tavily search |
| 📖 Wikipedia | Restricted to en.wikipedia.org |

Multiple sources can be selected simultaneously; results are merged and deduplicated.

### Document handling

- URLs ending in `.pdf` or returning `Content-Type: application/pdf` are extracted with **pypdf**
- HTML pages are extracted with **trafilatura** (clean-text extraction), with a plain-HTML fallback
- Each fetched document is chunked (800 chars / 100 overlap) and embedded with `text-embedding-3-small`
- FAISS indices are merged in-place as documents are added; removing a document triggers a full rebuild

## Setup

```bash
git clone <this-repo>
cd statt-eu-ai-act-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

Required keys:

- `OPENAI_API_KEY` — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- `TAVILY_API_KEY` — document search (free tier: 1,000 searches/month)

## Run

```bash
streamlit run app.py
# or
python app.py
```

Open http://localhost:8501

## Deploy — Streamlit Community Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io → **New app** → pick repo / branch `main` / `app.py`
3. **Advanced settings → Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."
   ```
4. Click **Deploy** (~2 min)

No pre-built index — the vector store is built on demand from whatever documents the user loads.

## Project layout

```
.
├── app.py                      Streamlit UI (search · preview · load · Q&A)
├── requirements.txt
├── README.md
├── REPORT.md
├── .env.example
├── .gitignore
├── runtime.txt / .python-version
├── .streamlit/config.toml
├── agent/
│   ├── agent.py               Topic-expert ReAct agent factory
│   ├── tools.py               make_search_tool() — FAISS-scoped search
│   └── rag.py                 build_doc_index() + persistent index helpers
└── data/
    ├── ingest.py
    └── docs/                  Legacy EU AI Act corpus (not used in main flow)
```
