"""One-shot script to (re)build the Chroma vector store.

Usage:
    python -m data.ingest
"""

from dotenv import load_dotenv

from agent.rag import DEFAULT_DOCS_DIR, DEFAULT_PERSIST_DIR, build_index


def main() -> None:
    load_dotenv()
    print(f"Reading documents from: {DEFAULT_DOCS_DIR}")
    print(f"Persisting index to:    {DEFAULT_PERSIST_DIR}")
    vs = build_index()
    count = vs._collection.count()  # type: ignore[attr-defined]
    print(f"Done. {count} chunks indexed.")


if __name__ == "__main__":
    main()
