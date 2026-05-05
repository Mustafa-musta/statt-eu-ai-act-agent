# Policy Research Agent

Agent-based RAG application for public policy research across Climate Change, Healthcare Reform, and Education Policy.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your keys
```

**Required keys** (in `.env` or Streamlit Cloud Secrets):
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Run

```bash
streamlit run app.py
```

## How it works

1. Pick a **topic** (Climate Change / Healthcare Reform / Education Policy) and **source** (Government, News & Media, Wikipedia — multi-select)
2. Search for documents — results are scoped to the selected topic
3. **Preview** a document before loading, then **Add to Analysis** (up to 6 documents)
4. Ask questions — the agent answers strictly from loaded document text, citing sources inline
5. If the answer is not in the documents it says so and prompts you to load more

## Deploy

Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → New app → set Secrets → Deploy.
