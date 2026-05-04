# EU AI Act Policy Agent

Agentic RAG application that answers questions about the **EU AI Act**
(Regulation (EU) 2024/1689) using a small curated corpus of policy
documents, with an optional live web-search fallback.

Built for the Statt Full-Stack ML Engineer take-home.

## Architecture

The agent is a LangGraph ReAct-style agent driving GPT-4o-mini, with
three tools. It decides per-query which tools to call — that is the
"agentic" piece, distinct from a fixed retrieve-then-generate pipeline.

```
                ┌─────────────────────────────┐
   user query ─►│  ReAct agent (gpt-4o-mini)  │── final answer + citations
                └──────────┬──────────────────┘
                           │ tool calls
        ┌──────────────────┼─────────────────────┐
        ▼                  ▼                     ▼
  search_policy_docs   lookup_document      web_search
  (Chroma + OpenAI     (full markdown       (Tavily, fallback
   embeddings)          file by name)        for current events)
```

The corpus is seven hand-curated markdown documents covering:
overview, risk classification, prohibited practices, high-risk systems,
GPAI obligations, penalties, and the implementation timeline. See
`data/docs/`.

## Setup

```bash
git clone <this-repo>
cd Statt_Full_Stack_TakeHome
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your keys
python -m data.ingest # build the vector store (one-time)
```

Required keys:

- `OPENAI_API_KEY` — sign up at https://platform.openai.com. Total cost
  for development plus reviewer demo is well under $1 with GPT-4o-mini.
- `TAVILY_API_KEY` — sign up at https://tavily.com. The free tier
  (1,000 searches/month) easily covers takehome usage. Optional: if you
  omit the key, the `web_search` tool is silently disabled.

## Run

CLI:

```bash
python cli.py
```

Streamlit UI:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. The UI shows a tool-call
trace under each answer so you can see exactly which tool the agent
chose.

## Deploy (free) — Streamlit Community Cloud

1. Push this repo to **GitHub** (it can be public; secrets stay out of
   the code thanks to `.gitignore` and `st.secrets`).
2. Go to https://share.streamlit.io and click **New app**.
3. Pick the repo, branch `main`, and entrypoint `app.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."
   ```

5. Click **Deploy**. After ~2 minutes you get a public
   `https://<your-app>.streamlit.app` URL.

The vector index is built lazily on the first request — no extra build
step is required on the deploy side. `chroma_db/` is gitignored, so each
fresh container builds it once and reuses it for the lifetime of the
container.

## Project layout

```
.
├── app.py                       Streamlit UI
├── cli.py                       CLI fallback
├── requirements.txt
├── README.md
├── REPORT.md                    1–2 page reflection
├── .env.example
├── .gitignore
├── .streamlit/config.toml
├── agent/
│   ├── agent.py                 LangGraph ReAct agent
│   ├── tools.py                 search_policy_docs, lookup_document, web_search
│   └── rag.py                   Chroma + OpenAI embeddings
└── data/
    ├── ingest.py                builds the vector index
    └── docs/                    7 EU AI Act markdown documents
```

## Example queries

- *"What practices are prohibited under Article 5?"* → corpus only.
- *"When do GPAI obligations apply?"* → corpus only.
- *"What's the maximum fine for breaching Article 5?"* → corpus only.
- *"How does the Act define a systemic-risk GPAI model?"* → corpus only.
- *"Which use cases count as high-risk under Annex III?"* → corpus only.
- *"Have there been any recent EU AI Act enforcement actions?"* →
  agent should call `web_search`.

## Notes

The application is a research tool, not legal advice. The agent is
prompted to add a one-line caveat when asked for legal advice.
