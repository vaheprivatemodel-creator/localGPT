"""
Qdrant-backed vector store with the same minimal contract LanceDB exposes
in ``rag_system.indexing.embedders``.

Runs Qdrant **embedded in-process** via ``QdrantClient(path=...)`` — no Docker,
no daemon. Persistent on-disk storage. Suitable for a single-host Mac Studio
deployment, which is the target environment.

Each chunk is stored as a point with:
    id      : deterministic uuid5 of chunk_id (so re-indexing is idempotent)
    vector  : np.float32 embedding
    payload :
        text         : str   (embedded text, may be enriched)
        chunk_id     : str
        document_id  : str
        chunk_index  : int
        user_id      : str|None (defence-in-depth for multi-tenant filtering)
        metadata     : str   (json-dumped full chunk dict, mirrors LanceDB schema)

The Qdrant store always indexes ``document_id`` and ``chunk_index`` as payload
indexes so that surrounding-chunk window queries are O(log n).
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient, models as qm


# Stable namespace so chunk_id → point uuid is reproducible across processes.
# Generated once via ``uuid.uuid4()``; do not change without bumping a migration.
_CHUNK_NAMESPACE = uuid.UUID("d0c47e6c-2c8d-4f67-9b9d-4a1f6e72b1c4")


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Deterministically map a chunk_id (string) to a Qdrant point id (uuid)."""
    return str(uuid.uuid5(_CHUNK_NAMESPACE, chunk_id))


class QdrantManager:
    """Thin wrapper around an embedded ``QdrantClient`` instance.

    Mirrors the role ``LanceDBManager`` plays for the LanceDB backend so call
    sites that previously held a ``LanceDBManager`` can be re-pointed without
    rewriting their flow.

    Embedded ``QdrantClient(path=...)`` holds an exclusive file lock on its
    data directory, so within a single process we MUST share one client per
    path. This class implements a per-path singleton: any two calls with the
    same resolved path get back the same underlying client/instance, which
    prevents the "Storage folder is already accessed by another instance of
    Qdrant client" error when multiple pipelines (indexing + retrieval)
    instantiate their own manager.
    """

    # path -> QdrantManager instance
    _instances: dict[str, "QdrantManager"] = {}
    _disable_singleton: bool = False  # tests can flip if needed

    def __new__(cls, path: str):
        full = os.path.abspath(path)
        if not cls._disable_singleton:
            existing = cls._instances.get(full)
            if existing is not None and getattr(existing, "client", None) is not None:
                return existing
        return super().__new__(cls)

    def __init__(self, path: str):
        full = os.path.abspath(path)
        # If this is the cached singleton, __init__ may be called again on the
        # already-initialised instance — short-circuit to avoid reopening the
        # client (which would deadlock on the file lock).
        if getattr(self, "_initialised", False) and self.path == full:
            return
        self.path = full
        os.makedirs(self.path, exist_ok=True)
        self.client = QdrantClient(path=self.path)
        self._initialised = True
        QdrantManager._instances[full] = self
        print(f"Qdrant (embedded) opened at: {self.path}")

    def close(self) -> None:
        """Release the embedded-mode file lock.

        The local ``QdrantClient`` holds an exclusive file lock on its data
        directory, so any code that needs to spawn a *second* client against
        the same path (e.g. delete a collection then re-open for indexing in
        a different module) must close the previous one first.
        """
        try:
            self.client.close()
        except Exception:
            pass
        # Forget the singleton so a subsequent constructor call gets a fresh
        # client. Callers that ``close()`` mid-run accept that they must
        # rebuild any retriever depending on the old client.
        self._initialised = False
        QdrantManager._instances.pop(self.path, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Do NOT auto-close on context exit: this instance may be shared with
        # other pipelines. Closing here would yank the lock out from under
        # them. The eval harness, which IS sure it is the sole owner, calls
        # close() explicitly.
        return False

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------
    def list_collections(self) -> List[str]:
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception as e:  # pragma: no cover - defensive
            print(f"⚠️  Qdrant list_collections failed: {e}")
            return []

    def has_collection(self, name: str) -> bool:
        return name in self.list_collections()

    def ensure_collection(self, name: str, vector_size: int) -> None:
        """Create the collection if it doesn't already exist.

        Uses cosine distance (the standard choice for sentence/qwen embeddings,
        all of which are L2-normalised). If a collection already exists with a
        different dimension we raise loudly rather than silently corrupting it.
        """
        if self.has_collection(name):
            info = self.client.get_collection(name)
            existing_size = info.config.params.vectors.size
            if existing_size != vector_size:
                raise ValueError(
                    f"Qdrant collection '{name}' has vector_size={existing_size} "
                    f"but caller wants {vector_size}. Delete the collection first."
                )
            return

        print(f"Creating Qdrant collection '{name}' (size={vector_size}, cosine)…")
        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )
        # Payload indexes for fast metadata filtering
        for field, schema in (
            ("document_id", qm.PayloadSchemaType.KEYWORD),
            ("chunk_id", qm.PayloadSchemaType.KEYWORD),
            ("chunk_index", qm.PayloadSchemaType.INTEGER),
            ("user_id", qm.PayloadSchemaType.KEYWORD),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=name, field_name=field, field_schema=schema
                )
            except Exception as e:  # pragma: no cover - field may already exist
                print(f"  (payload index '{field}' skipped: {e})")

    def delete_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
            print(f"Deleted Qdrant collection '{name}'.")
        except Exception as e:
            print(f"⚠️  Qdrant delete_collection('{name}') failed: {e}")

    # ------------------------------------------------------------------
    # Scrolling helpers (used by surrounding-chunks window expansion)
    # ------------------------------------------------------------------
    def scroll_by_filter(
        self, collection: str, payload_filter: qm.Filter, limit: int = 64
    ) -> List[Dict[str, Any]]:
        if not self.has_collection(collection):
            return []
        points, _ = self.client.scroll(
            collection_name=collection,
            scroll_filter=payload_filter,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        return [_point_to_dict(p) for p in points]


def _point_to_dict(point) -> Dict[str, Any]:
    """Normalise a Qdrant point into the dict shape RetrievalPipeline expects."""
    payload = point.payload or {}
    meta_raw = payload.get("metadata")
    if isinstance(meta_raw, str):
        try:
            metadata = json.loads(meta_raw)
        except Exception:
            metadata = {}
    elif isinstance(meta_raw, dict):
        metadata = meta_raw
    else:
        metadata = {}
    return {
        "chunk_id": payload.get("chunk_id"),
        "text": metadata.get("original_text", payload.get("text", "")),
        "document_id": payload.get("document_id"),
        "chunk_index": payload.get("chunk_index", -1),
        "metadata": metadata,
        "score": float(getattr(point, "score", 0.0) or 0.0),
        "user_id": payload.get("user_id"),
    }


class QdrantVectorIndexer:
    """Indexes (chunks, embeddings) into a Qdrant collection.

    API matches ``rag_system.indexing.embedders.VectorIndexer`` so the
    IndexingPipeline can swap them without changing its call shape.
    """

    def __init__(self, manager: QdrantManager):
        self.manager = manager

    def index(
        self,
        collection: str,
        chunks: List[Dict[str, Any]],
        embeddings,
        *,
        user_id: Optional[str] = None,
    ) -> None:
        # Defensive coupling: the upstream BatchProcessor swallows per-batch
        # failures (e.g. an MPS OOM) and returns FEWER embeddings than chunks.
        # Rather than raising and discarding the whole successful run, align
        # by truncating both to the minimum length and warn loudly.
        n_chunks = len(chunks)
        n_emb = len(embeddings) if embeddings is not None else 0
        if n_chunks != n_emb:
            print(
                f"⚠️  chunks/embeddings length mismatch ({n_chunks} chunks vs "
                f"{n_emb} embeddings). Truncating to {min(n_chunks, n_emb)} — "
                f"the remaining chunks will be skipped this run."
            )
            keep = min(n_chunks, n_emb)
            chunks = chunks[:keep]
            embeddings = embeddings[:keep]
        if not chunks:
            print("No chunks to index.")
            return

        vector_dim = int(embeddings[0].shape[0])
        self.manager.ensure_collection(collection, vector_dim)

        points: List[qm.PointStruct] = []
        skipped = 0
        for chunk, vector in zip(chunks, embeddings):
            arr = np.asarray(vector, dtype=np.float32)
            if not np.isfinite(arr).all():
                skipped += 1
                continue

            # Maintain LanceDB-compatible metadata shape so downstream code that
            # reads ``original_text`` from metadata keeps working unchanged.
            chunk_meta = chunk.get("metadata", {}) or {}
            if "original_text" not in chunk_meta:
                chunk_meta = {**chunk_meta, "original_text": chunk.get("text", "")}
            doc_id = chunk_meta.get("document_id") or chunk.get("document_id") or "unknown"
            chunk_idx = chunk_meta.get("chunk_index", chunk.get("chunk_index", -1))
            cid = chunk.get("chunk_id") or f"{doc_id}_{chunk_idx}"

            payload = {
                "text": chunk.get("text", "") or "",
                "chunk_id": cid,
                "document_id": doc_id,
                "chunk_index": int(chunk_idx) if chunk_idx is not None else -1,
                "user_id": user_id or chunk_meta.get("user_id"),
                # Mirror LanceDB schema: full chunk dict json-serialised
                "metadata": json.dumps({**chunk, "metadata": chunk_meta}, ensure_ascii=False),
            }
            points.append(
                qm.PointStruct(
                    id=chunk_id_to_point_id(cid),
                    vector=arr.tolist(),
                    payload=payload,
                )
            )

        if skipped:
            print(f"⚠️  Skipped {skipped} chunks with non-finite embeddings.")
        if not points:
            print("❌ No valid embeddings to index.")
            return

        # Upsert in batches to keep memory bounded for large ingests
        batch = 256
        total = 0
        for start in range(0, len(points), batch):
            chunk_points = points[start : start + batch]
            self.manager.client.upsert(
                collection_name=collection,
                points=chunk_points,
                wait=True,
            )
            total += len(chunk_points)
        print(f"✅ Upserted {total} points into Qdrant collection '{collection}'.")
