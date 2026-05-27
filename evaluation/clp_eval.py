"""
End-to-end CLP-derivative evaluation harness.

Runs the full local RAG stack against a target corpus and grades the
generated answer against ``clp_rubric.RUBRIC``.

Example (after moving the test docs into place)::

    cd /Users/vahemailyan/malyanLaw/localGPT
    source .venv/bin/activate
    python -m evaluation.clp_eval \
        --corpus /Users/vahemailyan/Downloads/tps-clp-test \
        --kb-id clp_eval_001 \
        --reindex

The harness:
1. (Optionally) re-indexes ``--corpus`` into the Qdrant collection
   ``kb_<kb-id>`` (default backend on this branch).
2. Issues the CLP-derivative test question plus a small set of derived
   stress-tests via the production RetrievalPipeline.
3. Scores each answer with the rubric.
4. Writes a JSON report to ``evaluation/results/`` and prints a summary.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import after sys.path tweak so we always pick up the repo's local rag_system
from rag_system.main import OLLAMA_CONFIG, PIPELINE_CONFIGS  # noqa: E402
from rag_system.utils.ollama_client import OllamaClient  # noqa: E402
from rag_system.vectorstore import qdrant_collection_name  # noqa: E402
from evaluation.clp_rubric import grade  # noqa: E402


PRIMARY_QUESTION = (
    "I need information on CLP applicability. The main asylum applicant is "
    "not barred by CLP but the derivative beneficiaries of his asylum have "
    "CLP issues. If the main applicant is granted asylum, can the other "
    "beneficiaries get asylum?"
)


# Derived stress-test questions (re-use the same retrieval/synthesis path).
# Each entry: (question, list of required substrings/regex-text that must
# appear in the answer for it to count as "correct"). These are intentionally
# light-weight; the full rubric only fires on the primary CLP question.
SECONDARY_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "tps_designation",
        "question": "Which countries are currently designated for Temporary Protected Status (TPS)?",
        "must_contain_any": ["TPS", "Temporary Protected Status"],
        "should_cite_doc": "Chart of TPS Countries",
    },
    {
        "id": "tps_initial_deadline",
        "question": "What is the registration window for an initial TPS application after a country is designated?",
        "must_contain_any": ["registration", "designation", "initial"],
        "should_cite_doc": "TPS Initial Request",
    },
    {
        "id": "clp_presumption_window",
        "question": "What is the rebuttable presumption period under the CLP rule and which regulation governs it?",
        "must_contain_any": ["208.33", "presumption"],
        "should_cite_doc": "9th circuit",
    },
    {
        "id": "ninth_cir_clp_holding",
        "question": "What did the Ninth Circuit hold about the CLP rule in the opinion in the knowledge base?",
        "must_contain_any": ["Ninth", "9th"],
        "should_cite_doc": "9th circuit",
    },
    {
        "id": "ead_extension_letter",
        "question": "What does the sample letter to an employer say about automatic extension of a TPS-based EAD?",
        "must_contain_any": ["EAD", "extension", "employer"],
        "should_cite_doc": "Automatic Extension of TPS EAD",
    },
]


def _collect_documents(corpus_dir: Path) -> List[str]:
    """Return absolute paths of supported PDF/DOCX files under corpus_dir."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    supported = (".pdf", ".docx")
    files = sorted(
        str(p) for p in corpus_dir.rglob("*") if p.suffix.lower() in supported and p.is_file()
    )
    return files


def build_pipeline_config(
    *,
    kb_id: str,
    qdrant_path: str,
    embedding_model: str = "nomic-embed-text",
    disable_enrichment: bool = False,
) -> Dict[str, Any]:
    """Clone the production "default" config and pin it to ``kb_<kb_id>``.

    The eval defaults to ``nomic-embed-text`` over the legacy
    ``Qwen/Qwen3-Embedding-0.6B`` HuggingFace model: empirically the HF model
    crashes with an MPS ``Invalid buffer size`` on batches of long enriched
    chunks (>35 GB intermediate buffer), silently dropping rows. Ollama's
    nomic-embed-text takes about a third of the time, runs out-of-process,
    and never hits the MPS allocator. The fallback to QwenEmbedder is one
    CLI flag away (``--embedding-model Qwen/Qwen3-Embedding-0.6B``).
    """
    import copy

    cfg = copy.deepcopy(PIPELINE_CONFIGS["default"])
    cfg["vector_backend"] = "qdrant"
    cfg["storage"]["qdrant_path"] = qdrant_path
    cfg["storage"]["text_table_name"] = qdrant_collection_name(kb_id)
    if cfg.get("retrieval", {}).get("late_chunking"):
        cfg["retrieval"]["late_chunking"]["lancedb_table_name"] = (
            qdrant_collection_name(kb_id, latechunk=True)
        )
    cfg["embedding_model_name"] = embedding_model
    # Reduce the embedding batch size to keep memory bounded even on Qwen/MPS.
    cfg.setdefault("indexing", {})["embedding_batch_size"] = 32
    if disable_enrichment:
        cfg.setdefault("contextual_enricher", {})["enabled"] = False
        # When enrichment is disabled, also drop late-chunk so we don't waste
        # cycles building a second view of identical text.
        if cfg.get("retrieval", {}).get("late_chunking"):
            cfg["retrieval"]["late_chunking"]["enabled"] = False
    return cfg


def _close_qdrant_holders(*objects) -> None:
    """Best-effort closure of any embedded-Qdrant clients these holders own,
    so the next phase can re-acquire the storage-folder lock.
    """
    for obj in objects:
        for attr in ("qdrant_manager", "manager", "_qdrant_manager"):
            mgr = getattr(obj, attr, None)
            if mgr is not None and hasattr(mgr, "close"):
                try:
                    mgr.close()
                except Exception:
                    pass


def run_indexing(cfg: Dict[str, Any], files: List[str]) -> None:
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline

    if not files:
        print("No files to index. Aborting.")
        return
    llm_client = OllamaClient(host=OLLAMA_CONFIG["host"])
    pipeline = IndexingPipeline(cfg, llm_client, OLLAMA_CONFIG)
    try:
        pipeline.run(file_paths=files)
    finally:
        _close_qdrant_holders(pipeline)
        del pipeline


def run_queries(cfg: Dict[str, Any], queries: List[str]) -> List[Dict[str, Any]]:
    """Open ONE long-lived RetrievalPipeline and reuse it for every query.

    Each ``RetrievalPipeline`` instance opens its own embedded Qdrant client,
    and the embedded client takes an exclusive file lock on ``qdrant_data/``.
    Instantiating per-query would fail with ``AlreadyLocked``.
    """
    from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline

    llm_client = OllamaClient(host=OLLAMA_CONFIG["host"])
    pipeline = RetrievalPipeline(cfg, llm_client, OLLAMA_CONFIG)
    results: List[Dict[str, Any]] = []
    try:
        for q in queries:
            start = time.time()
            out = pipeline.run(query=q, table_name=cfg["storage"]["text_table_name"])
            out["latency_s"] = round(time.time() - start, 2)
            results.append(out)
    finally:
        _close_qdrant_holders(pipeline)
        del pipeline
    return results


def secondary_score(item: Dict[str, Any], answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    answer_lc = (answer or "").lower()
    has_keyword = any(s.lower() in answer_lc for s in item.get("must_contain_any", []))
    cite_target = (item.get("should_cite_doc") or "").lower()
    cite_hit = False
    if cite_target:
        for c in retrieved or []:
            doc_id = (c.get("document_id") or "").lower()
            meta = c.get("metadata") or {}
            title = (meta.get("document_id") or meta.get("source") or "").lower()
            if cite_target in doc_id or cite_target in title:
                cite_hit = True
                break
    return {
        "id": item["id"],
        "question": item["question"],
        "has_keyword": has_keyword,
        "cited_expected_doc": cite_hit,
        "passed": has_keyword and (cite_hit or not cite_target),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CLP-derivative RAG evaluation")
    parser.add_argument("--corpus", type=str, required=False,
                        default=os.getenv("CLP_EVAL_CORPUS",
                                          "/Users/vahemailyan/Downloads/tps-clp-test"),
                        help="Directory containing PDF/DOCX test docs.")
    parser.add_argument("--kb-id", type=str, default="clp_eval_001",
                        help="Knowledge-base id; collection will be kb_<id>.")
    parser.add_argument("--qdrant-path", type=str, default="./qdrant_data")
    parser.add_argument("--reindex", action="store_true",
                        help="Wipe and re-ingest the corpus before evaluation.")
    parser.add_argument("--skip-secondary", action="store_true",
                        help="Only run the primary CLP question.")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text",
                        help="Embedding model. Pass an HF path (e.g. Qwen/Qwen3-Embedding-0.6B) "
                             "to use the legacy local Qwen embedder.")
    parser.add_argument("--no-enrichment", action="store_true",
                        help="Disable per-chunk contextual enrichment (saves ~25 min on 500-chunk corpora).")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus).expanduser().resolve()
    print(f"\n=== CLP-derivative evaluation ===")
    print(f"Corpus dir : {corpus_dir}")
    print(f"KB id      : {args.kb_id}  → collection {qdrant_collection_name(args.kb_id)}")
    print(f"Qdrant path: {args.qdrant_path}")
    print(f"Generation : {OLLAMA_CONFIG['generation_model']}")

    cfg = build_pipeline_config(
        kb_id=args.kb_id,
        qdrant_path=args.qdrant_path,
        embedding_model=args.embedding_model,
        disable_enrichment=args.no_enrichment,
    )
    print(f"Embedding  : {cfg['embedding_model_name']}")
    print(f"Enrichment : {'OFF' if args.no_enrichment else 'ON'}")

    if args.reindex:
        from rag_system.vectorstore.qdrant_store import QdrantManager
        from rag_system.vectorstore.bm25_sidecar import Bm25Sidecar

        coll = cfg["storage"]["text_table_name"]
        # Embedded Qdrant holds an exclusive file lock; scope this client tightly
        # so the IndexingPipeline can open its own client moments later.
        with QdrantManager(path=args.qdrant_path) as manager:
            manager.delete_collection(coll)
        bm25_dir = cfg["storage"].get("bm25_path", "./index_store/bm25")
        Bm25Sidecar(bm25_dir).delete(coll)
        files = _collect_documents(corpus_dir)
        if not files:
            print(f"❌ No PDF/DOCX files found in {corpus_dir}")
            return 2
        print(f"Indexing {len(files)} files into '{coll}'...")
        for f in files:
            print(f"  - {f}")
        run_indexing(cfg, files)

    # Build query list (one pipeline shared across all to keep the embedded
    # Qdrant file lock acquired exactly once).
    questions: List[str] = [PRIMARY_QUESTION]
    if not args.skip_secondary:
        questions.extend(item["question"] for item in SECONDARY_QUESTIONS)
    raw_results = run_queries(cfg, questions)

    # -------------------- Primary question --------------------
    primary_out = raw_results[0]
    answer = primary_out.get("answer", "")
    retrieved = primary_out.get("source_documents", [])
    print("\n--- Primary CLP-derivative question ---")
    print(f"Q: {PRIMARY_QUESTION}")
    print("\n--- Answer ---")
    print(answer)
    rubric_result = grade(answer, retrieved)
    print(f"\nScore: {rubric_result['earned']}/{rubric_result['max']} ({rubric_result['score']*100:.1f}%)")
    for d in rubric_result["details"]:
        marker = "✅" if d["passed"] else "❌"
        print(f"  {marker} {d['name']} (w={d['weight']}): {d['detail']}")

    # -------------------- Secondary questions --------------------
    secondary_results: List[Dict[str, Any]] = []
    if not args.skip_secondary:
        for item, sec in zip(SECONDARY_QUESTIONS, raw_results[1:]):
            print(f"\n--- Secondary: {item['id']} ---")
            print(f"Q: {item['question']}")
            ans = sec.get("answer", "")
            srcs = sec.get("source_documents", [])
            print("--- Answer ---")
            print(ans[:1500] + ("…" if len(ans) > 1500 else ""))
            scored = secondary_score(item, ans, srcs)
            marker = "✅" if scored["passed"] else "❌"
            print(f"{marker} keyword={scored['has_keyword']} cite_match={scored['cited_expected_doc']}")
            secondary_results.append({
                "id": item["id"],
                "question": item["question"],
                "answer": ans,
                "retrieved_docs": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "document_id": c.get("document_id"),
                        "score": c.get("score"),
                        "rerank_score": c.get("rerank_score"),
                    }
                    for c in srcs
                ],
                "score": scored,
                "latency_s": sec.get("latency_s"),
            })

    # -------------------- Report --------------------
    report = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "kb_id": args.kb_id,
        "corpus_dir": str(corpus_dir),
        "generation_model": OLLAMA_CONFIG["generation_model"],
        "primary": {
            "question": PRIMARY_QUESTION,
            "answer": answer,
            "latency_s": primary_out.get("latency_s"),
            "rubric": rubric_result,
            "retrieved_docs": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "document_id": c.get("document_id"),
                    "score": c.get("score"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in retrieved
            ],
        },
        "secondary": secondary_results,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"clp_{args.kb_id}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n📝 Report saved: {out_file}")

    primary_passed = rubric_result["score"] >= 0.75
    secondary_pass_count = sum(1 for r in secondary_results if r["score"]["passed"])
    print(
        f"\nSummary: primary {'PASS' if primary_passed else 'FAIL'} "
        f"({rubric_result['score']*100:.1f}%), "
        f"secondary {secondary_pass_count}/{len(secondary_results)}"
    )
    return 0 if primary_passed else 1


if __name__ == "__main__":
    sys.exit(main())
