# Approach, Challenges, and Improvements

## Policy area

I chose the **EU AI Act (Regulation (EU) 2024/1689)** for three reasons.
First, the primary text is authoritative and unambiguous, so retrieval
chunks have clean provenance. Second, it produces the kind of question
that benefits from retrieval rather than recall — specific articles,
dates, FLOPs thresholds, fine amounts — which is what a RAG system
should be good at. Third, it is recent enough that an occasional
fall-through to live web search is genuinely useful (for enforcement
news, Commission guidance, codes of practice), so the *agentic*
tool-routing earns its keep instead of being decorative.

## Architecture

The application is a LangGraph ReAct-style agent over `gpt-4o-mini`
with three tools:

1. `search_policy_docs(query)` — Chroma + `text-embedding-3-small`,
   `k=4`, top-similarity over a 7-document markdown corpus chunked at
   1000 characters / 150 overlap on a heading-aware splitter.
2. `lookup_document(filename)` — direct read of a full file when a
   semantic-search snippet is too narrow.
3. `web_search(query)` — Tavily, used as a fallback for queries the
   corpus cannot answer.

The agent loop runs until the model emits a final answer without
further tool calls. The system prompt instructs the model to ground
every claim, prefer the corpus, fall back to the web only when needed,
and cite sources inline (`[Source: …]` / `[Web: …]`). The Streamlit UI
exposes the tool-call trace in an expander so a reviewer can see
exactly which tool fired for each query — that is the visible
demonstration of agentic behaviour the spec asks for.

## Strengths

- **Multi-tool routing.** A vanilla RAG pipeline always retrieves; this
  agent retrieves only when retrieval helps. For "have there been any
  recent EU AI Act enforcement actions?" the agent skips the corpus and
  calls `web_search`. For "what is the maximum fine for breaching
  Article 5?" it calls only `search_policy_docs`.
- **Cited answers.** Every response includes inline source markers,
  traceable to specific files in `data/docs/` or to web URLs.
- **Cheap to run.** GPT-4o-mini at $0.15 / $0.60 per million input /
  output tokens, plus `text-embedding-3-small` at $0.02 per million,
  plus Tavily's free tier — total cost across development and demo is
  well under $1.
- **Free deployment path.** Streamlit Community Cloud runs the whole
  app at $0/month after the small per-query LLM cost.
- **Graceful degradation.** Missing `TAVILY_API_KEY`? The agent still
  works — the `web_search` tool is simply not registered.

## Limitations

- **Numerical fidelity.** Even with retrieval, the model occasionally
  paraphrases fines or thresholds. For legal use a stricter regime
  (extractive answering for numerical fields, or constrained generation
  with a regex schema for amounts and dates) would be safer.
- **No reranker.** Top-k similarity over seven documents is fine; at
  production scale a cross-encoder reranker (Cohere Rerank, or
  `bge-reranker-large`) would meaningfully improve precision against
  legal language where exact phrasing matters.
- **No evaluation harness.** I sanity-tested ~15 queries by hand. A
  production system needs a held-out QA set with metrics for
  faithfulness, answer relevance, and tool-selection accuracy (RAGAS
  or a custom rubric).
- **Single-turn agent calls.** The Streamlit UI keeps message history
  visually but each `agent.invoke` is independent. True multi-turn
  memory needs LangGraph's `MemorySaver` checkpointer keyed by
  thread id.
- **Brittle web fallback.** Tavily returns snippets, not full pages.
  For deeper web-grounded answers, fetching and re-chunking the top web
  result before passing it to the LLM would produce more reliable
  citations.
- **Static corpus.** The seven documents are a snapshot. Real-world use
  needs a re-ingest pipeline triggered by changes to authoritative
  sources (EUR-Lex, Commission guidance pages).

## Improvements for production

| Area | Change |
| --- | --- |
| Retrieval quality | Hybrid search (BM25 + dense, RRF), cross-encoder reranker |
| Evaluation | Continuously graded QA set; RAGAS faithfulness / answer relevance; tool-selection accuracy as a first-class metric |
| Observability | Langfuse or LangSmith for traces, latency, cost, prompt-version tracking |
| Caching | Semantic cache (e.g. GPTCache) — for a public chatbot the largest single cost saver |
| Guardrails | Off-topic detection, jailbreak resistance, refusal pattern for legal-advice queries |
| Streaming | Token streaming in Streamlit improves perceived latency at zero accuracy cost |
| Vector store | Migrate from local Chroma to pgvector or Pinecone with metadata filtering and per-document ACLs |
| Authn / authz | If extended to proprietary policy corpora, retrieval results must be filtered by user entitlements |
| Model tiering | Route legal/numerical questions to a stronger model (Claude Sonnet, GPT-4o); keep mini for routine retrieval — a small query classifier in front of the agent |
| Re-ingest | Scheduled pipeline that watches authoritative sources and re-chunks/embeds on change |
| Cost monitoring | Per-tenant token accounting and rate-limiting |

## Cost envelope

For the take-home itself (development plus reviewer demo, ~100 queries
at ~3K input / ~500 output tokens each): under $0.10 for LLM, free for
Tavily, free for Streamlit Cloud — well under $1 in total.

For a small production deployment (1,000 user queries/month with two
search calls each): ~$3–5/month at GPT-4o-mini, $0–$8/month for Tavily,
$0 for Streamlit Cloud — call it under $15/month all-in. Scaling to
100K queries/month is dominated by LLM costs (~$300/month at
GPT-4o-mini) and is the point at which a semantic cache + reranker
combination starts paying for itself.

## Time spent

Roughly 3.5 hours: 30 min corpus curation, 60 min RAG plus agent and
tools, 30 min Streamlit UI plus CLI, 30 min deployment configuration
and secrets handling, 30 min report and README, 30 min testing and
polish.
