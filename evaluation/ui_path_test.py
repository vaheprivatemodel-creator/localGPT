"""End-to-end UI-path test for localGPT.

Exercises the same HTTP chain a real browser exercises:
    POST http://localhost:3000/api/auth/login            (Next.js → backend :8002)
    POST http://localhost:3000/api/sessions              (Next.js → backend :8002)
    POST http://localhost:3000/api/sessions/<id>/indexes/<idx>   (Next.js → backend :8002)
    POST http://localhost:3000/api/stream                (Next.js → rag-api :8001)

Streams the SSE response, prints stage-by-stage timings, the final answer
and the list of source documents.  Pass/fail report at the end.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

import requests

UI = "http://localhost:3000"
ADMIN_EMAIL = "admin@firm.com"
ADMIN_PASSWORD = "changeme123"
QUERY = (
    "I need information on CLP applicability. The main asylum applicant "
    "is not barred by CLP but the derivative beneficiaries of his asylum "
    "have CLP issues. If the main applicant is granted asylum, can the "
    "other beneficiaries get asylum?"
)
CLP_INDEX_NAME_KEYWORD = "CLP"   # matches "CLP eval corpus (pre-indexed)"

# ─────────────────────────── helpers ───────────────────────────
def must(resp: requests.Response, label: str) -> dict[str, Any]:
    if not resp.ok:
        print(f"❌ {label}: HTTP {resp.status_code}  body={resp.text[:300]}")
        sys.exit(1)
    try:
        return resp.json()
    except Exception:
        return {}


def login() -> str:
    r = requests.post(f"{UI}/api/auth/login",
                      json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
                      timeout=10)
    data = must(r, "login")
    token = data.get("token")
    if not token:
        print("❌ no token in login response:", data); sys.exit(1)
    print(f"✅ login ok  token={token[:8]}…")
    return token


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def list_models(token: str):
    r = requests.get(f"{UI}/api/models", headers=auth(token), timeout=10)
    data = must(r, "list models")
    raw = data.get("generation_models", []) or data.get("models", []) or []
    gens = [m.get("id") if isinstance(m, dict) else m for m in raw]
    print(f"✅ /models returns {len(gens)} generation models (first 5: {gens[:5]})")


def find_clp_index(token: str) -> tuple[str, str]:
    r = requests.get(f"{UI}/api/indexes", headers=auth(token), timeout=10)
    data = must(r, "list indexes")
    for idx in data.get("indexes", []):
        if CLP_INDEX_NAME_KEYWORD.lower() in (idx.get("name") or "").lower():
            print(f"✅ found index '{idx.get('name')}'  id={idx['id'][:8]}…  "
                  f"vector_table={idx.get('vector_table_name')}")
            return idx["id"], idx.get("vector_table_name") or ""
    print("❌ no CLP index found. Available:",
          [i.get("name") for i in data.get("indexes", [])])
    sys.exit(1)


def new_session(token: str) -> str:
    r = requests.post(f"{UI}/api/sessions", headers=auth(token),
                      json={"title": "ui_test"}, timeout=10)
    data = must(r, "create session")
    sid = data.get("session", {}).get("id") or data.get("session_id") or data.get("id")
    if not sid:
        print("❌ no session id in response:", data); sys.exit(1)
    print(f"✅ session created  id={sid[:8]}…")
    return sid


def link_index(token: str, sid: str, idx_id: str):
    r = requests.post(f"{UI}/api/sessions/{sid}/indexes/{idx_id}",
                      headers=auth(token), timeout=10)
    must(r, "link index")
    print(f"✅ linked index {idx_id[:8]}… → session {sid[:8]}…")


def stream_chat(token: str, sid: str) -> dict[str, Any]:
    """Returns dict {answer, sources, model, stages, total_s, error}."""
    body = {
        "query": QUERY,
        "session_id": sid,
        # Intentionally DO NOT send table_name — this is the new front-end
        # behaviour after the fix.
        "force_rag": True,
        "compose_sub_answers": True,
        "ai_rerank": True,
        "context_expand": True,
        "verify": False,
        "retrieval_k": 20,
        "context_window_size": 1,
        "reranker_top_k": 10,
        "search_type": "hybrid",
        "dense_weight": 0.7,
        # Do not pin a model from the client — the rag-api will default to
        # qwen2.5:14b per the fix.
    }
    print("⏳ POST /api/stream  …  (this is the exact browser path)")
    t0 = time.time()
    stages: list[tuple[str, float]] = []
    answer = ""
    sources: list[Any] = []
    last_evt = None
    err: str | None = None
    with requests.post(f"{UI}/api/stream", headers={**auth(token),
                                                    "Content-Type": "application/json",
                                                    "Accept": "text/event-stream"},
                       data=json.dumps(body), stream=True, timeout=180) as r:
        if not r.ok:
            print(f"❌ stream HTTP {r.status_code}  body={r.text[:300]}")
            sys.exit(1)
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            try:
                evt = json.loads(raw[5:].strip())
            except Exception:
                continue
            etype = evt.get("type")
            last_evt = etype
            now = time.time() - t0
            if etype in ("analyze", "decomposition", "retrieval_started",
                         "retrieval_done", "rerank_started", "rerank_done",
                         "context_expand_started", "context_expand_done",
                         "final_answer", "single_query_result"):
                stages.append((etype, now))
                print(f"   {now:6.2f}s  ▸ {etype}")
            elif etype == "token":
                tok = (evt.get("data") or {}).get("text") or ""
                answer += tok
                # progress dots
                if len(answer) % 200 < len(tok):
                    print(f"   {now:6.2f}s  📝 streamed {len(answer)} chars")
            elif etype == "sub_query_result":
                stages.append(("sub_query_result", now))
                sub = evt.get("data") or {}
                preview = (sub.get("answer") or "")[:120].replace("\n", " ")
                print(f"   {now:6.2f}s  ▸ sub_query_result  '{sub.get('query', '')[:60]}…'  → '{preview}…'")
            elif etype == "complete":
                stages.append(("complete", now))
                data = evt.get("data") or {}
                answer = data.get("answer") or answer
                sources = data.get("source_documents") or sources
                print(f"   {now:6.2f}s  ✅ complete")
                break
            elif etype == "error":
                err = (evt.get("data") or {}).get("error", "unknown")
                print(f"   {now:6.2f}s  ❌ error  {err}")
                break
            else:
                stages.append((etype, now))
    return {"answer": answer, "sources": sources, "stages": stages,
            "total_s": time.time() - t0, "error": err, "last": last_evt}


def grade(answer: str, sources: list[Any]) -> dict[str, Any]:
    """Simple 6-point rubric for the CLP question."""
    a = answer.lower()
    checks = {
        "mentions_clp_or_terrorism_related_inadmissibility":
            ("clp" in a) or ("terrorism-related" in a) or ("inadmissibility" in a),
        "addresses_derivative_beneficiaries":
            "derivative" in a or "beneficiar" in a,
        "principal_grant_does_not_cure_derivative_bar":
            re.search(r"(cannot|does not|do not|no entitlement)", a) is not None
            and ("derivative" in a or "beneficiar" in a),
        "discusses_waiver_or_exemption":
            "waiver" in a or "exemption" in a or "exempt" in a or "discretion" in a,
        "discusses_separate_eligibility_path":
            "follow-to-join" in a or "independent" in a or "i-730" in a
            or "their own asylum" in a or "individual application" in a,
        "has_citations":
            len(sources) >= 1 and bool(re.search(r"\[[Ss]\d+\]|\[\d+\]|p\.\s*\d+", answer)),
    }
    passed = sum(1 for v in checks.values() if v)
    return {"checks": checks, "passed": passed, "total": len(checks)}


def main():
    print("=" * 72)
    print("localGPT UI-path end-to-end test")
    print("=" * 72)
    tok = login()
    list_models(tok)
    idx_id, vt = find_clp_index(tok)
    sid = new_session(tok)
    link_index(tok, sid, idx_id)
    result = stream_chat(tok, sid)

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"total_time    : {result['total_s']:.1f}s")
    print(f"last_event    : {result['last']}")
    print(f"answer_len    : {len(result['answer'])} chars")
    print(f"num_sources   : {len(result['sources'])}")
    if result["error"]:
        print(f"error         : {result['error']}")

    print("\n— STAGE TIMELINE —")
    for s, t in result["stages"]:
        print(f"  {t:6.2f}s   {s}")

    print("\n— ANSWER —")
    print(result["answer"][:2400])
    if len(result["answer"]) > 2400:
        print(f"... (+{len(result['answer'])-2400} more chars)")

    print("\n— SOURCES (first 5) —")
    for s in (result["sources"] or [])[:5]:
        if isinstance(s, dict):
            md = s.get("metadata") or {}
            print(f"  • file={md.get('document_id') or md.get('source')}  "
                  f"page={md.get('page_number') or md.get('page')}  "
                  f"chunk_id={s.get('chunk_id') or md.get('chunk_id')}")
        else:
            print(f"  • {str(s)[:160]}")

    g = grade(result["answer"], result["sources"] or [])
    print("\n— RUBRIC (6 pts) —")
    for k, v in g["checks"].items():
        print(f"  {'✅' if v else '❌'}  {k}")
    print(f"\nSCORE: {g['passed']}/{g['total']}")
    print(f"SESSION_ID: {sid}")


if __name__ == "__main__":
    main()
