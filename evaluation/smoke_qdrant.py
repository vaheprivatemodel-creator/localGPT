"""Quick smoke test for the new Qdrant vectorstore + BM25 sidecar.

Run with:
    cd /Users/vahemailyan/malyanLaw/localGPT
    source .venv/bin/activate
    python -m evaluation.smoke_qdrant
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure repo root on path when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_system.vectorstore import qdrant_collection_name, legacy_to_qdrant_name
from rag_system.vectorstore.qdrant_store import QdrantManager, QdrantVectorIndexer
from rag_system.vectorstore.bm25_sidecar import Bm25Sidecar
from rag_system.vectorstore.qdrant_retriever import QdrantMultiVectorRetriever


class FakeEmbedder:
    """Deterministic 32-d toy embedder for the smoke test."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.rng = np.random.default_rng(seed=1234)
        self._cache: dict[str, np.ndarray] = {}

    def create_embeddings(self, texts):
        out = []
        for t in texts:
            if t not in self._cache:
                # Deterministic per-text vector via hashing
                seed = abs(hash(t)) % (2**32)
                rng = np.random.default_rng(seed=seed)
                v = rng.normal(size=self.dim).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-9
                self._cache[t] = v
            out.append(self._cache[t])
        return np.stack(out)


def assert_eq(a, b, msg):
    if a != b:
        raise AssertionError(f"{msg}: expected {b!r}, got {a!r}")
    print(f"  ✓ {msg}")


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="qdrant_smoke_")
    try:
        qpath = os.path.join(workdir, "qdrant_data")
        bm25dir = os.path.join(workdir, "bm25")

        assert_eq(qdrant_collection_name("abc-123"), "kb_abc-123", "collection name new")
        assert_eq(qdrant_collection_name("abc-123", latechunk=True), "kb_abc-123_lc", "collection name lc")
        assert_eq(legacy_to_qdrant_name("text_pages_abc-123"), "kb_abc-123", "legacy translate")
        assert_eq(legacy_to_qdrant_name("text_pages_abc-123_lc"), "kb_abc-123_lc", "legacy translate lc")
        assert_eq(legacy_to_qdrant_name("kb_already-new"), "kb_already-new", "pass-through")

        manager = QdrantManager(path=qpath)
        indexer = QdrantVectorIndexer(manager)
        bm25 = Bm25Sidecar(bm25dir)

        # Synthetic chunks
        chunks = [
            {
                "chunk_id": "doc1_0",
                "text": "Asylum derivative beneficiaries follow-to-join under INA § 208(b)(3).",
                "metadata": {
                    "document_id": "INA.pdf",
                    "chunk_index": 0,
                    "original_text": "Asylum derivative beneficiaries follow-to-join under INA § 208(b)(3).",
                    "user_id": "u1",
                },
            },
            {
                "chunk_id": "doc1_1",
                "text": "Circumvention of Lawful Pathways (CLP) bar at 8 C.F.R. § 208.33.",
                "metadata": {
                    "document_id": "INA.pdf",
                    "chunk_index": 1,
                    "original_text": "Circumvention of Lawful Pathways (CLP) bar at 8 C.F.R. § 208.33.",
                    "user_id": "u1",
                },
            },
            {
                "chunk_id": "doc2_0",
                "text": "Lopez-Cardona v. Garland addressed CLP applicability in the Ninth Circuit.",
                "metadata": {
                    "document_id": "9th-cir.pdf",
                    "chunk_index": 0,
                    "original_text": "Lopez-Cardona v. Garland addressed CLP applicability in the Ninth Circuit.",
                    "user_id": "u1",
                },
            },
        ]

        embedder = FakeEmbedder(dim=32)
        embeddings = embedder.create_embeddings([c["text"] for c in chunks])

        coll = qdrant_collection_name("smoke-idx")
        indexer.index(coll, chunks, embeddings, user_id="u1")
        bm25.add(coll, chunks)

        # Stored?
        info = manager.client.get_collection(coll)
        assert_eq(info.points_count, 3, "qdrant point count")

        retriever = QdrantMultiVectorRetriever(
            qdrant_manager=manager,
            text_embedder=embedder,
            bm25_sidecar=bm25,
            user_id="u1",
        )
        hits = retriever.retrieve(
            text_query="CLP derivative beneficiary asylum",
            table_name=coll,
            k=3,
        )
        print(f"  → {len(hits)} hits returned")
        for h in hits:
            print(f"    - {h['chunk_id']} score={h['score']:.3f} bm25={h.get('bm25')}")

        if not hits:
            raise AssertionError("Retriever returned no hits")
        top_ids = {h["chunk_id"] for h in hits}
        # We expect doc1_1 (the CLP-tagged chunk) to be in the top results.
        assert "doc1_1" in top_ids, f"Expected doc1_1 in top hits, got {top_ids}"
        print("  ✓ hybrid retrieval surfaces CLP chunk")

        # Surrounding chunks
        surrounding = retriever.get_surrounding_chunks(coll, "INA.pdf", 0, window_size=1)
        ids = [s["chunk_id"] for s in surrounding]
        assert ids == ["doc1_0", "doc1_1"], f"surrounding window ids unexpected: {ids}"
        print("  ✓ surrounding-chunks window correct")

        # Cross-user isolation (defence-in-depth)
        retriever_other_user = QdrantMultiVectorRetriever(
            qdrant_manager=manager,
            text_embedder=embedder,
            bm25_sidecar=bm25,
            user_id="u2",
        )
        empty = retriever_other_user.retrieve(
            text_query="CLP derivative beneficiary asylum",
            table_name=coll,
            k=3,
        )
        # BM25 sidecar has no user filter yet so empty isn't guaranteed to be empty;
        # only the Qdrant leg is filtered. Still, vector hits should be 0 for u2.
        # That means score breakdown should lack any vec_score_norm.
        for h in empty:
            assert "vec_score_norm" not in h, f"User isolation breach: {h}"
        print("  ✓ Qdrant user_id filter blocks cross-user vector hits")

        print("\n🎉 Qdrant smoke test passed.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
