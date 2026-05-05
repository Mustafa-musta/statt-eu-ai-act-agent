# Report — Policy Research Agent

## Policy Area

Three domains: **Climate Change**, **Healthcare Reform**, and **Education Policy** — the exact examples given in the brief. These were chosen because each has a healthy mix of government reports (often PDFs), news articles, and Wikipedia content, making multi-source retrieval genuinely useful.

## Approach

The application is a **dynamic fetch-and-RAG** agent. Rather than pre-collecting a fixed corpus, the user discovers and loads documents at runtime:

1. **Document search** — Tavily searches the web, scoped to the active topic and optionally restricted to government domains (unfccc.int, who.int, ed.gov, …), news outlets, or Wikipedia.
2. **Content extraction** — HTML pages are cleaned with `trafilatura`; PDFs (common for policy reports) are extracted with `pypdf`.
3. **Indexing** — each document is chunked (800 chars / 100 overlap), embedded with `text-embedding-3-small`, and merged into a shared in-memory FAISS index. Up to 6 documents can be combined.
4. **Agent Q&A** — a LangGraph ReAct agent (`gpt-4o-mini`, temperature 0) has one tool: `search_document(query)`, which retrieves the top-5 passages from the combined index. The system prompt gives the agent a topic-specific expert persona (e.g. climate policy analyst, health systems researcher) and instructs it to answer **only from retrieved text**. If the answer is absent it says *"Not found — load more documents."*

## Strengths

- **No static corpus needed** — any publicly accessible document can be loaded on demand.
- **PDF support** — handles the majority of authoritative policy documents.
- **Strict grounding** — the agent is explicitly forbidden from using training knowledge, reducing hallucination on fact-sensitive policy content.
- **Multi-source, multi-document** — government, news, and Wikipedia results can be combined in one index and queried together.

## Limitations

- **Session-only state** — loaded documents and the FAISS index are lost on page refresh.
- **No reranker** — top-k similarity is adequate; a cross-encoder reranker would improve precision over heterogeneous documents.
- **Single-turn memory** — each `agent.invoke` is independent; multi-turn context requires LangGraph's `MemorySaver`.
- **Rebuild on removal** — removing one of six loaded documents triggers full re-embedding of the remainder (FAISS has no delete API).

## Improvements for Production

| Area | Change |
|---|---|
| Persistence | Save FAISS index + document texts to S3/GCS per user session |
| Retrieval | Hybrid BM25 + dense search with cross-encoder reranker |
| Evaluation | RAGAS faithfulness / answer-relevance on a held-out QA set |
| Observability | LangSmith traces for latency, cost, and prompt-version tracking |
| Streaming | Token streaming to reduce perceived latency |
| OCR | pytesseract fallback for scanned PDFs that pypdf cannot extract |
