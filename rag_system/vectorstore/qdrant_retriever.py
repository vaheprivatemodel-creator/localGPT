"""
Qdrant-backed hybrid retriever — drop-in replacement for
``rag_system.retrieval.retrievers.MultiVectorRetriever``.

Returned chunk dicts intentionally mirror the LanceDB retriever's shape so
``RetrievalPipeline`` can consume them without code changes:

    {
        "chunk_id": str,
        "text":     str,                      # original_text from metadata
        "score":    float,                    # fused score (higher = better)
        "bm25":     float | None,
        "_distance": float | None,            # cosine distance, smaller = closer
        "document_id": str,
        "chunk_index": int,
        "metadata": dict,
    }
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from rag_system.vectorstore.qdrant_store import QdrantManager
from rag_system.vectorstore.bm25_sidecar import Bm25Sidecar


logger = logging.getLogger(__name__)


def _normalise(values: List[float]) -> List[float]:
    """Min-max normalise to [0,1]; constant vectors collapse to 0.5."""
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class QdrantMultiVectorRetriever:
    """Hybrid (vector + BM25) retriever over a single Qdrant collection."""

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        text_embedder,
        bm25_sidecar: Bm25Sidecar,
        *,
        fusion_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ):
        self.manager = qdrant_manager
        self.text_embedder = text_embedder
        self.bm25 = bm25_sidecar
        self.fusion_config = fusion_config or {
            "method": "linear",
            "bm25_weight": 0.5,
            "vec_weight": 0.5,
        }
        # Optional defence-in-depth filter so user A cannot read user B's
        # points even if they accidentally share a collection.
        self.user_id = user_id

        @lru_cache(maxsize=256)
        def _embed_single(q: str):
            return self.text_embedder.create_embeddings([q])[0]

        self._embed_single = _embed_single

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_filter(self):
        if not self.user_id:
            return None
        from qdrant_client import models as qm

        return qm.Filter(
            must=[
                qm.FieldCondition(
                    key="user_id",
                    match=qm.MatchValue(value=self.user_id),
                )
            ]
        )

    def _vector_search(self, collection: str, vector, limit: int) -> List[Dict[str, Any]]:
        if not self.manager.has_collection(collection):
            return []
        try:
            hits = self.manager.client.search(
                collection_name=collection,
                query_vector=list(map(float, vector)),
                limit=limit,
                query_filter=self._build_filter(),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.debug("Qdrant vector search failed on '%s': %s", collection, e)
            return []

        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            meta_raw = payload.get("metadata")
            try:
                metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            except Exception:
                metadata = {}
            if isinstance(metadata, dict) and isinstance(metadata.get("metadata"), dict):
                # Nested shape produced by QdrantVectorIndexer
                inner = metadata.get("metadata", {})
                # Merge top-level fields like text/chunk_id into the inner metadata
                metadata = {**metadata, **inner}
            # Qdrant cosine score: higher = better, in [0,1]
            vec_score = float(getattr(hit, "score", 0.0) or 0.0)
            cosine_distance = max(0.0, 1.0 - vec_score)
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "text": metadata.get("original_text", payload.get("text", "")),
                    "_distance": cosine_distance,
                    "vec_score": vec_score,
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index", -1),
                    "metadata": metadata,
                }
            )
        return results

    def _bm25_search(self, collection: str, query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            return self.bm25.query(collection, query, k=limit)
        except Exception as e:
            logger.debug("BM25 sidecar query failed on '%s': %s", collection, e)
            return []

    # ------------------------------------------------------------------
    # Public API (matches MultiVectorRetriever.retrieve signature)
    # ------------------------------------------------------------------
    def retrieve(
        self,
        text_query: str,
        table_name: str,
        k: int,
        reranker=None,
    ) -> List[Dict[str, Any]]:
        # ``table_name`` is the legacy parameter name from the LanceDB retriever.
        # In the new world it's a Qdrant collection. We accept either form.
        from rag_system.vectorstore import legacy_to_qdrant_name

        collection = legacy_to_qdrant_name(table_name)
        logger.debug(
            "Qdrant hybrid retrieve: query='%s' coll='%s' k=%s", text_query, collection, k
        )
        if reranker is not None:
            logger.debug("Ignoring legacy LanceDB reranker arg in Qdrant retriever.")

        # Fetch ~2x candidates from each leg so fusion has room to swap order
        per_leg_k = max(k, 1) * 2
        q_vec = self._embed_single(text_query)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            vec_future = executor.submit(self._vector_search, collection, q_vec, per_leg_k)
            bm25_future = executor.submit(self._bm25_search, collection, text_query, per_leg_k)
            vec_hits = vec_future.result()
            bm25_hits = bm25_future.result()

        # ------------------------------------------------------------------
        # Linear-fusion: min-max normalise each leg, then weighted sum.
        # ------------------------------------------------------------------
        w_bm25 = float(self.fusion_config.get("bm25_weight", 0.5))
        w_vec = float(self.fusion_config.get("vec_weight", 0.5))

        vec_scores = _normalise([h["vec_score"] for h in vec_hits])
        bm25_scores = _normalise([h["bm25"] for h in bm25_hits])

        merged: Dict[str, Dict[str, Any]] = {}
        for h, s in zip(vec_hits, vec_scores):
            cid = h.get("chunk_id") or json.dumps(h, sort_keys=True, default=str)
            entry = merged.setdefault(cid, {**h, "score": 0.0})
            entry["score"] += w_vec * s
            entry["vec_score_norm"] = s
        for h, s in zip(bm25_hits, bm25_scores):
            cid = h.get("chunk_id") or json.dumps(h, sort_keys=True, default=str)
            entry = merged.setdefault(cid, {**h, "score": 0.0, "_distance": None})
            entry["score"] += w_bm25 * s
            entry["bm25"] = h.get("bm25")

        ranked = sorted(merged.values(), key=lambda d: d.get("score", 0.0), reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Surrounding-chunks helper (used by RetrievalPipeline context window)
    # ------------------------------------------------------------------
    def get_surrounding_chunks(
        self,
        collection: str,
        document_id: str,
        chunk_index: int,
        window_size: int,
    ) -> List[Dict[str, Any]]:
        from qdrant_client import models as qm
        from rag_system.vectorstore import legacy_to_qdrant_name

        collection = legacy_to_qdrant_name(collection)
        if document_id is None or chunk_index is None or chunk_index < 0:
            return []

        start_index = max(0, chunk_index - window_size)
        end_index = chunk_index + window_size
        must = [
            qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
            qm.FieldCondition(
                key="chunk_index",
                range=qm.Range(gte=start_index, lte=end_index),
            ),
        ]
        if self.user_id:
            must.append(qm.FieldCondition(key="user_id", match=qm.MatchValue(value=self.user_id)))
        payload_filter = qm.Filter(must=must)

        rows = self.manager.scroll_by_filter(collection, payload_filter, limit=window_size * 4 + 4)
        rows.sort(key=lambda r: r.get("chunk_index", 0))
        return rows
