"""
BM25 sidecar index for Qdrant collections.

Qdrant does not natively provide BM25 / full-text scoring. LanceDB did
(via Tantivy FTS), so when we migrated to Qdrant we lost the BM25 leg of
the hybrid retrieval. This sidecar restores it without adding a separate
service:

  * One pickle per Qdrant collection: ``<bm25_dir>/<collection>.pkl``
  * On disk schema (versioned):
        {
            "version": 1,
            "tokens":     List[List[str]],     # token streams, one per chunk
            "chunk_ids":  List[str],
            "payloads":   List[Dict[str, Any]] # what to return on a hit
        }
  * Tokeniser: simple, language-agnostic, lower-case alphanumerics + a tiny
    legal-aware regex for citations like ``8 C.F.R. § 208.13(c)(2)``.
  * Rebuilds the underlying ``BM25Okapi`` from tokens on every load.

This is intentionally simple. We trade a small amount of memory for full
control over scoring and the ability to keep the dense and sparse views
exactly in sync (the indexer adds to both atomically).
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi


_CITATION_TOKEN = re.compile(
    r"""
    (?:
      \b\d+\s*[Uu]\.?\s*[Ss]\.?\s*[Cc]\.?(?:\s*§)?\s*\d+[a-zA-Z\d\.\(\)]* |   # 8 USC 1158
      \b\d+\s*[Cc]\.?\s*[Ff]\.?\s*[Rr]\.?(?:\s*§)?\s*\d+[a-zA-Z\d\.\(\)]* |   # 8 C.F.R. § 1208.13(c)
      \bINA\s*§?\s*\d+[a-zA-Z\d\.\(\)]*                                       # INA § 208(b)(3)
    )
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> List[str]:
    """Tokenise for BM25 with a tiny legal-citation booster.

    Citations are emitted as a single normalised token *and* their component
    words, so both "section 208" and the exact string "8 C.F.R. § 208.13" land
    in the index. That helps recall on legal queries without exploding the
    vocabulary.
    """
    if not text:
        return []
    text_low = text.lower()
    cites = [m.group(0) for m in _CITATION_TOKEN.finditer(text)]
    cite_tokens = [re.sub(r"\s+", "", c.lower()) for c in cites]
    words = re.findall(r"[a-z0-9§]+", text_low)
    return words + cite_tokens


class Bm25Sidecar:
    """Per-collection BM25 store kept in sync with Qdrant by the indexer."""

    SCHEMA_VERSION = 1

    def __init__(self, bm25_dir: str):
        self.bm25_dir = os.path.abspath(bm25_dir)
        os.makedirs(self.bm25_dir, exist_ok=True)
        # Cache the loaded BM25 model in memory per collection
        self._cache: Dict[str, Tuple[BM25Okapi, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # File-path helpers
    # ------------------------------------------------------------------
    def _path(self, collection: str) -> str:
        # Collection names from the migration are guaranteed safe (kb_<uuid>),
        # but normalise defensively to avoid path traversal if someone passes
        # an attacker-controlled string.
        safe = re.sub(r"[^A-Za-z0-9_\-\.]", "_", collection)
        return os.path.join(self.bm25_dir, f"{safe}.pkl")

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------
    def _load_state(self, collection: str) -> Dict[str, Any]:
        path = self._path(collection)
        if not os.path.exists(path):
            return {
                "version": self.SCHEMA_VERSION,
                "tokens": [],
                "chunk_ids": [],
                "payloads": [],
            }
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        if state.get("version") != self.SCHEMA_VERSION:
            # Future: handle migrations. For now, start fresh.
            return {
                "version": self.SCHEMA_VERSION,
                "tokens": [],
                "chunk_ids": [],
                "payloads": [],
            }
        return state

    def _save_state(self, collection: str, state: Dict[str, Any]) -> None:
        path = self._path(collection)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        self._cache.pop(collection, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, collection: str, chunks: List[Dict[str, Any]]) -> None:
        """Append chunks to the on-disk BM25 store.

        If a chunk_id already exists, its tokens/payload are replaced so that
        re-indexing the same document does not produce duplicate hits.
        """
        if not chunks:
            return
        state = self._load_state(collection)
        existing_idx: Dict[str, int] = {cid: i for i, cid in enumerate(state["chunk_ids"])}

        for ch in chunks:
            cid = ch.get("chunk_id")
            if not cid:
                continue
            text = ch.get("text") or ""
            metadata = ch.get("metadata", {}) or {}
            # Prefer original_text for sparse search so enrichment prefixes
            # (e.g. "Context: …") don't pollute the BM25 vocabulary.
            sparse_text = metadata.get("original_text", text)
            tokens = _tokenize(sparse_text)
            payload = {
                "chunk_id": cid,
                "text": sparse_text,
                "document_id": metadata.get("document_id") or ch.get("document_id"),
                "chunk_index": metadata.get("chunk_index", ch.get("chunk_index", -1)),
                "metadata": metadata,
            }
            if cid in existing_idx:
                i = existing_idx[cid]
                state["tokens"][i] = tokens
                state["payloads"][i] = payload
            else:
                state["tokens"].append(tokens)
                state["chunk_ids"].append(cid)
                state["payloads"].append(payload)
        self._save_state(collection, state)
        print(f"🔎 BM25 sidecar '{collection}': {len(state['chunk_ids'])} chunks indexed.")

    def query(
        self, collection: str, text_query: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        if not text_query or k <= 0:
            return []

        if collection in self._cache:
            bm25, state = self._cache[collection]
        else:
            state = self._load_state(collection)
            if not state["tokens"]:
                return []
            bm25 = BM25Okapi(state["tokens"])
            self._cache[collection] = (bm25, state)

        q_tokens = _tokenize(text_query)
        if not q_tokens:
            return []
        scores = bm25.get_scores(q_tokens)
        if not len(scores):
            return []

        # Top-k by descending score, drop zeros
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Dict[str, Any]] = []
        for i in order:
            s = float(scores[i])
            if s <= 0.0:
                break
            payload = dict(state["payloads"][i])
            payload["bm25"] = s
            out.append(payload)
            if len(out) >= k:
                break
        return out

    def delete(self, collection: str) -> None:
        path = self._path(collection)
        if os.path.exists(path):
            os.remove(path)
        self._cache.pop(collection, None)
