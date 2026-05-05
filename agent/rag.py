"""Retrieval layer: FAISS vector store + OpenAI embeddings.

Two modes:
- build_doc_index(): in-memory index from any fetched document text (dynamic RAG)
- build_index() / get_vectorstore(): persistent index from the local corpus files
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = ROOT / "data" / "docs"
DEFAULT_PERSIST_DIR = ROOT / "faiss_db"
EMBED_MODEL = "text-embedding-3-small"


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


def build_doc_index(text: str, title: str, source_url: str) -> FAISS:
    """Build an in-memory FAISS index from a single fetched document."""
    doc = Document(page_content=text, metadata={"title": title, "source": source_url})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents([doc])
    return FAISS.from_documents(chunks, embedding=_embeddings())


def build_index(
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> FAISS:
    docs_dir = Path(docs_dir)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    raw_docs: List[Document] = loader.load()
    if not raw_docs:
        raise RuntimeError(f"No markdown documents found under {docs_dir}.")

    for d in raw_docs:
        d.metadata["filename"] = Path(d.metadata.get("source", "")).name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    vs = FAISS.from_documents(chunks, embedding=_embeddings())
    vs.save_local(str(persist_dir))
    return vs


def get_vectorstore(persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> FAISS:
    return FAISS.load_local(
        str(persist_dir),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def index_exists(persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> bool:
    p = Path(persist_dir)
    return (p / "index.faiss").exists()
