# LocalGPT — Vahe CLP-eval patches (v2 + v3)

This `localgpt-v2` working tree carries patches applied during a two-session
evaluation of `PromtEngineer/localGPT @ localgpt-v2` against the
`9th circuit addending including CLP.pdf` document.

It is **not** intended to be upstreamed as-is. The patches mix three concerns:

1. **v2 quality patches** — derived from a baseline eval that found the stock
   build couldn't recognize legal acronyms (CLP), gave 0 inline citations, and
   reported a meaningless `[Confidence: 100%]` on wrong answers. Five fixes were
   applied; see "v2 patches" below.
2. **v3 quality patches** — derived from a v2 follow-up eval that found the v2
   answer still mis-attributed a CFR citation (`1208.13(c)` content tagged as
   `1208.33(a)(3)`) and that the option-parsing in the build endpoint silently
   ignored snake_case keys from the README. Two more fixes were applied.
3. **Local-environment shims** — the developer's machine already runs another
   service on port 8000 (`/opt/vahe-lawyer/rag_api`), so the backend was moved
   to 8002 and the frontend's hard-coded `API_BASE_URL` was retargeted.

Detailed before/after numbers and commentary live in the user's primary repo:

- `vahe-milen-lianna-david/reports/localgpt-v2-clp-eval.md` (baseline)
- `vahe-milen-lianna-david/reports/localgpt-v2-clp-eval-v2.md` (v2)
- `vahe-milen-lianna-david/reports/localgpt-v2-clp-eval-v3.md` (v3)

---

## v2 patches

| File | What changed | Why |
| --- | --- | --- |
| `rag_system/main.py` | `OLLAMA_CONFIG["generation_model"]` → `qwen2.5:14b` (overridable via `GENERATION_MODEL` env var). Session-create endpoint also defaults to the same model. | qwen3:8b couldn't reliably paraphrase a multi-paragraph CFR rule; qwen2.5:14b can. |
| `rag_system/retrieval/query_transformer.py` | Added "Acronym Expansion" section to the `QueryDecomposer` system prompt with a starter dictionary (CLP, CAT, PSG, BIA, IJ, EOIR, DHS, INA, C.F.R.). | Without expansion the decomposer treats "CLP" as a stop-word and retrieves nothing relevant. |
| `rag_system/pipelines/retrieval_pipeline.py` | Snippets are now passed to the generator as `[S1] (chunk_id=...) <text>`. Synthesis prompt mandates `[S#]` on every factual claim and encourages inline hybrid cites like `[S3, 8 C.F.R. § 1208.33(a)(2)(ii)]`. Acronym instruction repeated here so the *generator* (not just the decomposer) resolves it. | Inline citations are required for auditability; the [S#] index makes per-claim cite cross-checks possible. |
| `rag_system/agent/loop.py` | Same acronym + `[S#]` instructions in the composer prompt. Added a grounded verifier footer (`verifier=<verdict> \| grounded=<y/n> \| cited <n>/<m> \| verifier_score=N%`) replacing the opaque `[Confidence: N%]`. Counts citations by regex on the answer body. | Stops the LLM from inflating its own confidence; surfaces warnings like `NOT_GROUNDED`, `VERIFIER_NOT_SUPPORTED`. |
| `backend/server.py` | `PORT = int(os.environ.get("BACKEND_PORT", "8002"))` — was hard-coded 8000. | Port 8000 belongs to `/opt/vahe-lawyer/rag_api` on this machine. |
| `run_system.py`, `src/lib/api.ts` | Wired through the 8002 port change. | — |

---

## v3 patches (on top of v2)

| File | What changed | Why |
| --- | --- | --- |
| `rag_system/agent/loop.py` | Added module-level `hard_groundedness_check(answer, source_documents) -> dict`. Catches `PHANTOM_SOURCE` (out-of-range `[S#]`), `MISCITED` (`[S1, REG]` where `REG` doesn't literally appear in S1), `UNGROUNDED_CITE` (any CFR/INA reg in the answer that doesn't appear in any retrieved chunk), and `NO_INLINE_CITATIONS`. Aggressive normalization: strip whitespace + periods + `§` and match by numeric section signature so `8 C.F.R.` vs `8 C.F.R` vs `8 CFR` never false-positives. Replaced the v2 footer block to use the deterministic verdict as the headline (`deterministic_check=pass\|FAIL`) and demote the LLM verifier to a supplementary line (`verifier(llm)=...`). | The v2 verifier was an LLM grading its own work, and it never noticed a literal mis-attribution. The deterministic check is unspoofable and catches the exact class of bug v2 missed. |
| `backend/server.py` | Replaced the `handle_build_index` option-parsing block with an `_opt(opts, *aliases, default, cast)` helper that accepts both `chunk_size`/`chunkSize`, `latechunk`/`enableLatechunk`, etc. Parse errors and unknown keys now `print` warnings instead of being silently dropped. | The README's example payload used `chunk_size` but the server only read `chunkSize`, so every build silently used the default 512. |

---

## Headline before/after

| | Baseline | v2 patched | **v3** |
| --- | --- | --- | --- |
| Q1 (verbatim "CLP") | 2/10 | 6/10 | **7/10** |
| Q2 (expanded CLP) | 7/10 | 8.5/10 | **9/10** |
| Q2 mis-attribution | n/a | present (`[S1, 1208.33(a)(3)]` for `1208.13(c)`) | **gone, deterministic_check=pass** |
| Composite (Q1+Q2) | 45% | 73% | **80%** |
| `chunk_size` honored from snake_case POST | — | ❌ ignored | ✅ honored |
| Q2 latency | 262 s | 51 s | **44 s** |

---

## Known-remaining defects (after v3)

1. **Synthesis still under-utilises retrieved chunks on Q1.** The most relevant CFR
   subsection (`1208.33(a)(2)(ii)`, family-with-whom-traveling exception) was in
   the retrieved set on multiple runs but didn't make it into the answer. The
   deterministic check correctly flags the answer as `FAIL` (model leaned on
   parametric knowledge for `INA § 208(b)(3)(A)`, which has 0 occurrences in
   Q1's 7 retrieved chunks), but doesn't *fix* the underlying retrieval/synthesis
   gap.
2. **No retry on `deterministic_check=FAIL`.** "Fix #2" from the v2→v3 handoff
   was intentionally skipped — it requires a small plumbing change to
   `retrieval_pipeline.run()` to accept an extra prompt addendum. Expected lift
   on Q1 is +1 to +2 points at the cost of ~80 s extra latency.
3. **The deterministic check is purely textual.** It can't catch *semantic*
   mis-attribution where the cite is plausibly nearby in the chunk (model
   paraphrases sentence A but cites the regulation from sentence B in the same
   chunk). Would need claim-level alignment, probably via NLI. Out of scope for
   this patch series.
4. **Verifier still emits `verifier_score=N%`** alongside the deterministic
   verdict. It's now clearly labelled `verifier(llm)=...`, but a downstream
   caller that regex-greps `\d+%` could still be misled. Clean follow-up:
   drop the percentage and only emit `verifier(llm)={supported,partially_supported,not_supported}`.

---

## How to reset / rebuild

```bash
# Stack restart (uses port 8002 for backend, 8001 for rag-api)
cd /Users/vahemailyan/malyanLaw/localGPT
source .venv/bin/activate
NODE_OPTIONS="--no-experimental-webstorage" python run_system.py --mode dev
```

```bash
# Rebuild v3 index (snake_case POST should be honored — fix #4 verification)
ID=$(curl -sS -X POST http://localhost:8002/indexes -H "Content-Type: application/json" \
  -d '{"name":"CLP-v3","description":"snake_case test"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["index_id"])')
curl -sS -X POST "http://localhost:8002/indexes/${ID}/upload" \
  -F "files=@/path/to/9th circuit addending including CLP.pdf" >/dev/null
curl -sS -X POST "http://localhost:8002/indexes/${ID}/build" \
  -H "Content-Type: application/json" \
  -d '{"chunk_size":768,"chunk_overlap":96,"latechunk":true,"enable_enrich":true}'
```

Expect `chunk_size: 768` and `latechunk: true` in the response.
