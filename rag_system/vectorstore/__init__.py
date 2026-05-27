"""
Pluggable vector-store backends.

The current production backend is LanceDB (see ``rag_system.indexing.embedders``
and ``rag_system.retrieval.retrievers``). This package introduces a Qdrant
implementation that mirrors the same minimal contract, so the
indexing/retrieval pipelines can switch backends via the
``vector_backend`` configuration key.

Contract (used by IndexingPipeline + RetrievalPipeline):

    class VectorStoreManager:
        def __init__(self, path: str): ...
        def ensure_collection(self, name: str, vector_size: int) -> None: ...
        def delete_collection(self, name: str) -> None: ...
        def list_collections(self) -> list[str]: ...

    class VectorIndexer:
        def __init__(self, manager: VectorStoreManager): ...
        def index(self, collection: str, chunks: list[dict], embeddings: np.ndarray) -> None: ...

    class MultiVectorRetriever:
        def retrieve(self, text_query: str, table_name: str, k: int,
                     reranker=None) -> list[dict]: ...
        # returned dicts must include: chunk_id, text, score, document_id,
        # chunk_index, metadata.

Per-knowledge-base isolation is enforced by collection-name convention:
    legacy LanceDB:  text_pages_<idx_id> / text_pages_<idx_id>_lc
    new    Qdrant:   kb_<idx_id>          / kb_<idx_id>_lc

Helpers below make naming explicit.
"""
from __future__ import annotations

LEGACY_COLLECTION_PREFIX = "text_pages_"
NEW_COLLECTION_PREFIX = "kb_"
LATECHUNK_SUFFIX = "_lc"


def qdrant_collection_name(index_id: str, *, latechunk: bool = False) -> str:
    """Return the canonical Qdrant collection name for a given knowledge-base id."""
    base = f"{NEW_COLLECTION_PREFIX}{index_id}"
    return f"{base}{LATECHUNK_SUFFIX}" if latechunk else base


def legacy_to_qdrant_name(legacy_table_name: str) -> str:
    """Translate a legacy LanceDB table name to the equivalent Qdrant collection.

    Handles both ``text_pages_<id>`` and ``text_pages_<id>_lc`` shapes, plus the
    pass-through case where the caller already passed a ``kb_*`` name.
    """
    if not legacy_table_name:
        return legacy_table_name
    if legacy_table_name.startswith(NEW_COLLECTION_PREFIX):
        return legacy_table_name
    lc = legacy_table_name.endswith(LATECHUNK_SUFFIX)
    core = legacy_table_name[: -len(LATECHUNK_SUFFIX)] if lc else legacy_table_name
    if core.startswith(LEGACY_COLLECTION_PREFIX):
        idx_id = core[len(LEGACY_COLLECTION_PREFIX):]
        return qdrant_collection_name(idx_id, latechunk=lc)
    return legacy_table_name
