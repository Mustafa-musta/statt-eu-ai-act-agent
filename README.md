# Policy Research Agent

Agent-based RAG application for public policy research across Climate Change, Healthcare Reform, and Education Policy.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your keys
```

**Required keys:**
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Run

```bash
streamlit run app.py
```

## How to use

1. Pick a **topic** and one or more **sources** (Government, News & Media, Wikipedia) in the sidebar
2. Type a query and click **Find Documents** — results are scoped to the selected topic
3. Click **👁️ Preview** to inspect a document before committing
4. Click **📥 Add to Analysis** to embed it (up to 6 documents)
5. Ask questions in the chat — answers come strictly from loaded document text
6. If the answer is not found, the agent says so and prompts you to load more documents

## Deploy

Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → New app → set Secrets → Deploy.

---

## Code walkthrough

### `agent/rag.py` — Retrieval layer

**`_embeddings()`**
Returns an `OpenAIEmbeddings` object using `text-embedding-3-small`. Every chunk of text is passed through this model to produce a 1536-dimension vector capturing its semantic meaning.

**`build_doc_index(text, title, source_url)`**
The core dynamic RAG function. Takes raw document text, wraps it in a LangChain `Document` with metadata, splits it into chunks with `RecursiveCharacterTextSplitter` (chunk size 800, overlap 100 — prefers `\n\n` → `\n` → `. ` break points so context is not lost at arbitrary cuts), embeds every chunk via the OpenAI API, and stores the vectors in an in-memory FAISS index. Nothing is written to disk — the index lives only in RAM for the session.

**`build_index()` / `get_vectorstore()` / `index_exists()`**
Legacy functions from the original static corpus. Read `.md` files from `data/docs/`, chunk, embed, and persist to `faiss_db/`. Not used in the main app flow but kept for the CLI.

---

### `agent/tools.py` — Search tool

**`make_search_tool(vs)`**
A factory that takes a FAISS index and returns a LangChain `StructuredTool` permanently bound to that index via closure. The inner `_search(query)` converts the query to a vector, finds the 5 nearest chunks by cosine similarity, and formats them as `[Source: title]\n<chunk text>` blocks. The closure means each time the combined index is updated, a new tool is created that captures the latest state. The tool's `description` string is what the LLM reads to decide when to call it.

---

### `agent/agent.py` — Agent brain

**System prompt — three layers**

1. `_EXPERT_PROMPTS` — topic-specific persona. Climate Change: senior climate policy analyst fluent in IPCC reports, NDCs, carbon markets. Healthcare Reform: health systems researcher who knows DRG payment, UHC, pharma pricing. Education Policy: comparative education specialist who knows PISA, Title I, achievement gaps. This shapes how the model frames retrieved content.
2. `_BASE_RULES` — hard constraints: always call `search_document` first, never use training knowledge, if not found say the exact "Not found" string, cite every claim with `[Source: ...]`.
3. `_build_system_prompt(topic)` — joins persona + multi-doc note + rules into one string.

**`build_doc_agent(vs, topic, model)`**
Creates `ChatOpenAI` (GPT-4o-mini, temperature 0 — fully deterministic) and calls `create_react_agent(llm, tools, prompt)`. This is a LangGraph function that compiles a state graph implementing the ReAct loop: the LLM sees the system prompt and conversation, decides whether to call a tool or give a final answer, if it calls a tool LangGraph executes it and feeds the result back, and this repeats until the LLM stops calling tools and produces the answer.

---

### `app.py` — Streamlit UI

**Startup sequence**
1. `load_dotenv()` reads `.env` into environment variables
2. `_hydrate_secrets()` copies Streamlit Cloud secrets into `os.environ` — Streamlit Cloud does not use `.env` files
3. API key guard — if `OPENAI_API_KEY` is missing, `st.stop()` halts all further execution

**Constants**
`TOPICS` maps each topic to its icon, colour, trusted government domains, and default query. `SOURCE_META` maps each source mode to its Tavily configuration — Government uses topic-specific `gov_domains`, Wikipedia has fixed domains, News has no restriction.

**Session state**
Streamlit reruns the entire script on every interaction. Session state persists values across reruns:
- `active_topic` / `active_sources` — current selections
- `search_results` — last Tavily search results
- `loaded_docs` — list of `{title, url, source}` for every loaded document
- `doc_texts` — `url → (text, title)` used to rebuild the index on document removal
- `vs` — combined in-memory FAISS index
- `agent` — LangGraph agent (rebuilt when documents or topic change)
- `messages` — full chat history
- `previews` — cached preview text per URL (avoids re-fetching)

**`_scoped_query(query, topic)`**
Prepends the topic name to the user's query if not already present — "policy report 2024" becomes "Climate Change policy report 2024" — so Tavily results stay on topic even with a generic query.

**`_multi_source_search(query, topic, sources)`**
Loops over all selected sources. For each one, builds a Tavily kwargs dict with optional `include_domains`. Searches each source separately, deduplicates results by URL using a `seen` set, tags each result with `_source` for the badge in the UI, caps total at 6.

**`_fetch_text(url)`**
Three-layer extraction pipeline:
1. URL ends in `.pdf` → `requests` download + `pypdf` page extraction
2. Otherwise `trafilatura.fetch_url()` + `trafilatura.extract()` — strips ads, navigation, boilerplate; also catches silent PDF redirects via `%PDF` magic bytes check
3. Fallback: `requests` GET + minimal HTML parser that skips `<script>`/`<style>` tags and joins all remaining text nodes

**`_add_document(text, title, url, source)`**
Calls `build_doc_index()` to create a new FAISS index for the document, stores raw text in `doc_texts`, then either sets `vs` to the new index (first document) or calls `vs.merge_from(new_vs)` which merges the new vectors into the combined index in-place. Rebuilds the agent with the updated index.

**`_remove_document(url)`**
Removes the entry from `doc_texts` and `loaded_docs`, then rebuilds the entire combined FAISS index from scratch by iterating remaining entries and merging them. Full rebuild is necessary because FAISS has no delete API — individual vectors cannot be removed.

**UI layout**
Two columns: left for document discovery (search, result cards, preview), right for Q&A chat. Result cards are rendered as styled HTML with title, URL, snippet, and a coloured source badge. Preview button fetches and caches text, toggling an expander with the first 2000 characters. Load button is disabled if already loaded or at the 6-document limit.

**Chat loop**
`st.chat_input()` blocks until submission. The message is appended to `messages`, then `agent.invoke({"messages": [("user", prompt)]})` runs the LangGraph ReAct loop synchronously — internally it may call `search_document` one or more times before returning the final answer. Tool calls are extracted from intermediate messages and shown in a collapsible expander.
