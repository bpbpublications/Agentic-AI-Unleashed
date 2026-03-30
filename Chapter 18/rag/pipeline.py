# rag/pipeline.py
# ============================================================
# RAG Pipeline: FAISS Index + Retrieval
# Agentic AI Unleashed — Appendix Demo
#
# Builds an in-memory FAISS vector index from the clinical
# corpus at startup. No external database is required —
# everything lives in process memory, making the demo fully
# self-contained and portable.
#
# Design note: In production you would replace FAISS with a
# persistent vector store (Pinecone, Weaviate, pgvector, etc.)
# and add chunking, metadata filtering, and re-ranking.
# ============================================================

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from rag.corpus import CLINICAL_DOCUMENTS

# Module-level singleton — index is built once at import time
_vector_store: FAISS | None = None


def build_index() -> FAISS:
    """
    Embed all clinical document chunks and build the FAISS index.
    Called once at application startup.
    """
    print("[RAG] Building FAISS index from clinical corpus...")

    documents = [
        Document(
            page_content=doc["content"],
            metadata={
                "id":     doc["id"],
                "title":  doc["title"],
                "source": doc["source"],
                "url":    doc["url"],
            },
        )
        for doc in CLINICAL_DOCUMENTS
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_documents(documents, embeddings)

    print(f"[RAG] Index built — {len(documents)} documents indexed.")
    return store


def get_vector_store() -> FAISS:
    """
    Return the singleton FAISS index, building it on first call.
    Thread-safe for the single-process demo context.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = build_index()
    return _vector_store


def retrieve(query: str, k: int = 3, max_l2_distance: float = 0.75) -> list[dict]:
    """
    Retrieve the top-k most relevant clinical document chunks
    for a given query string, filtered by L2 distance threshold.

    Uses similarity_search_with_score to obtain distances, then
    filters out results whose L2 distance exceeds max_l2_distance.
    Returns a score field in each result dict (lower = more similar).

    Args:
        query:            The natural language search query.
        k:                Number of top candidates to retrieve (default 3).
        max_l2_distance:  Maximum allowable L2 distance (default 0.75).
                          Results above this threshold are discarded.
    """
    store = get_vector_store()
    # Returns list of (Document, score) tuples; score is L2 distance (lower = better)
    results_with_scores = store.similarity_search_with_score(query, k=k)

    filtered = [
        {
            "content":  doc.page_content,
            "title":    doc.metadata.get("title", "Unknown"),
            "source":   doc.metadata.get("source", "Unknown"),
            "url":      doc.metadata.get("url", ""),
            "score":    float(score),  # cast numpy.float32 → Python float for JSON serialization
        }
        for doc, score in results_with_scores
        if score <= max_l2_distance
    ]

    if not filtered:
        print(f"[RAG] Warning: no results passed the L2 distance threshold ({max_l2_distance}) "
              f"for query: '{query[:80]}'")

    return filtered
