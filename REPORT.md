# Approach, Challenges, and Improvements

## Policy areas

The agent covers three policy domains — **Climate Change**, **Healthcare Reform**, and
**Education Policy** — chosen because they each have a rich mix of official government
documents, peer-reviewed research, and live news, making multi-source retrieval genuinely
useful rather than decorative.

Each topic has a distinct expert persona baked into the agent's system prompt, so the
model frames retrieved content through the right conceptual vocabulary (e.g. NDCs and
carbon budgets for climate; DRG payment and UHC for healthcare; PISA and achievement gaps
for education).

## Architecture

The application is a **dynamic fetch-and-RAG** system built on LangGraph and FAISS.
There is no pre-built corpus: every document is fetched at query time from the live web.

### Flow

1. **Topic + source selection** — the user picks one of three topics and one or more
   source modes (Government & Official, News & Media, Wikipedia).
2. **Scoped document search** — a Tavily search is issued with the topic name
   prepended to the user's query and, for government mode, domain-restricted to
   authoritative sites (unfccc.int, who.int, ed.gov, …).
3. **Preview** — the user can fetch and inspect the first 2 000 characters of any
   result before committing.
4. **Load** — clicking *Add to Analysis* fetches the full document (PDF via pypdf,
   HTML via trafilatura), chunks it (800 chars / 100 overlap), embeds it with
   `text-embedding-3-small`, and merges the new FAISS index into the shared index.
   Up to six documents can be loaded simultaneously.
5. **Q&A** — a LangGraph ReAct agent with one tool (`search_document`) retrieves the
   top-5 passages and answers strictly from retrieved text, citing the source document
   inline. If the answer is not present it says so and prompts the user to load more
   documents.

### Agent

- Model: `gpt-4o-mini` (temperature 0)
- Tool: `search_document(query)` — FAISS similarity search over the combined index, k=5
- System prompt: topic-expert persona + strict document-only rule
  ("treat yourself as a reader who has only ever seen these documents")
- Not-found response: *"Not found in the loaded documents. I can search more documents —
  use the Find Documents panel to add additional sources."*

## Strengths

- **Dynamic corpus.** No corpus curation or re-ingest pipeline needed — the user
  controls exactly which documents are in scope.
- **Multi-source, multi-document.** Government reports, news articles, and Wikipedia
  can be searched simultaneously and combined into a single shared index.
- **PDF support.** pypdf handles government reports and research papers that are
  served as PDFs, which are the majority of authoritative policy documents.
- **Topic-expert personas.** The agent interprets retrieved text through domain
  vocabulary (carbon markets, DRG payment, PISA) rather than as a generic reader.
- **Strict grounding.** The agent is forbidden from using training knowledge to fill
  gaps, which reduces hallucination in a fact-sensitive policy context.
- **Preview before load.** Users can inspect document content before committing an
  embedding API call.
- **Source provenance.** Every result card and loaded-document badge shows which
  source mode it came from, maintaining clear attribution.

## Limitations

- **Tavily snippet depth.** Tavily returns clean snippets but not always the complete
  document. Very long PDFs may be truncated by pypdf if the server enforces download
  limits.
- **No reranker.** Top-k similarity is adequate for focused queries; a cross-encoder
  reranker would improve precision when six heterogeneous documents are loaded.
- **Session-only memory.** Loaded documents and the FAISS index live in Streamlit
  session state and are lost on page refresh. A persistent user workspace would
  require a server-side store.
- **Single-turn agent calls.** Each `agent.invoke` is independent. True multi-turn
  memory across conversation turns requires LangGraph's `MemorySaver` checkpointer.
- **No evaluation harness.** Answers are validated manually. A production system needs
  a QA set with RAGAS faithfulness / answer-relevance metrics.
- **Rebuild cost on removal.** Removing one document from the six-document index
  triggers a full re-embedding of all remaining documents (no FAISS delete API).

## Improvements for production

| Area | Change |
|---|---|
| Retrieval quality | Hybrid search (BM25 + dense, RRF fusion), cross-encoder reranker |
| Evaluation | Continuously graded QA set; RAGAS faithfulness / relevance; tool-selection accuracy |
| Observability | LangSmith or Langfuse for traces, latency, cost, and prompt-version tracking |
| Persistence | Save loaded documents and FAISS index to object storage (S3 / GCS) per user session |
| Streaming | Token streaming in Streamlit for lower perceived latency |
| Caching | Semantic cache (GPTCache) — largest single cost saver for repeated queries |
| PDF quality | OCR fallback (pytesseract) for scanned PDFs that pypdf cannot extract |
| Model tiering | Route numerical / legal questions to a stronger model; keep mini for routine retrieval |
| Authn / authz | If extended to proprietary corpora, filter retrieval results by user entitlements |
| More topics | Modular TOPICS config makes adding new domains (defence, housing, trade) trivial |

## Cost envelope

Per-session cost (loading 3 documents, ~20 queries):

| Component | Cost |
|---|---|
| Embeddings (3 docs × ~50 chunks × 800 tokens) | ~$0.002 |
| LLM (20 queries × ~3K input / 500 output tokens) | ~$0.006 |
| Tavily (a few searches) | Free tier |
| **Total per session** | **< $0.01** |

For 1 000 sessions/month: ~$10 all-in at current GPT-4o-mini pricing.

## Time spent

Roughly 5 hours: 45 min architecture design, 60 min agent + RAG layer, 60 min Streamlit
UI redesign (multi-doc, multi-source, preview), 30 min PDF support and source scoping,
30 min deployment and secrets handling, 30 min report and README, 15 min testing.
