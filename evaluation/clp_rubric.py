"""
CLP-derivative eligibility rubric.

Each check is a callable that returns ``(passed: bool, detail: str)`` given the
LLM answer and the list of retrieved chunks. The rubric mixes deterministic
regex/citation checks with a few light "did the answer mention this concept"
checks. It deliberately avoids LLM-as-judge so the eval is reproducible.

Acceptance criteria (mapped from the user's CLP-derivative question):

1.  Answer mentions derivative asylum / follow-to-join.
2.  Cites INA § 208(b)(3) (or its U.S.C. mirror, 8 U.S.C. § 1158(b)(3)).
3.  Cites 8 C.F.R. § 208.21 (or § 1208.21) — the derivative procedure.
4.  Acknowledges CLP / "Circumvention of Lawful Pathways" by name.
5.  Cites the CLP rule itself (8 C.F.R. § 208.33 or § 1208.33).
6.  Reaches the correct legal conclusion: derivatives CAN obtain derivative
    asylum when the principal is granted, even if the derivative would
    independently face a CLP bar.
7.  Calls out the principal-vs-derivative distinction explicitly.
8.  Mentions qualifying-derivative limits (spouse / unmarried child < 21)
    OR references INA § 208(b)(3)(A).
9.  Includes inline ``[S#]`` snippet labels (the synthesizer prompt requires
    them; if absent the answer is ungrounded).
10. All citations in the answer appear verbatim in at least one retrieved
    chunk (no hallucinated regs / cases).
11. Surfaces a Ninth-Circuit hook from the corpus when one is present.
12. Lists a Knowledge-Base Gap when any of the controlling authorities above
    are missing from the retrieved snippets (per the synthesizer prompt).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


@dataclass
class CheckResult:
    name: str
    passed: bool
    weight: float
    detail: str


# ---------------------------------------------------------------------------
# Regex toolbox
# ---------------------------------------------------------------------------
RE_INA_208_B3 = re.compile(r"INA\s*§?\s*208\s*\(\s*b\s*\)\s*\(\s*3\s*\)", re.IGNORECASE)
RE_USC_1158_B3 = re.compile(
    r"8\s*U\.?\s*S\.?\s*C\.?\s*§?\s*1158\s*\(\s*b\s*\)\s*\(\s*3\s*\)", re.IGNORECASE
)
RE_CFR_208_21 = re.compile(
    r"8\s*C\.?\s*F\.?\s*R\.?\s*§?\s*1?208\.21", re.IGNORECASE
)
RE_CFR_208_33 = re.compile(
    r"8\s*C\.?\s*F\.?\s*R\.?\s*§?\s*1?208\.33", re.IGNORECASE
)
RE_CLP_FULL = re.compile(
    r"circumvent\w*\s+(of\s+)?lawful\s+pathways", re.IGNORECASE
)
RE_CLP_ACR = re.compile(r"\bCLP\b")
RE_DERIVATIVE = re.compile(r"derivative\s+(?:asylum|benefic|status|appl)", re.IGNORECASE)
RE_FOLLOW_TO_JOIN = re.compile(r"follow[- ]to[- ]join", re.IGNORECASE)
RE_INLINE_CITE = re.compile(r"\[S\d+(?:\s*,[^\]]+)?\]")
RE_NINTH_CIR = re.compile(r"\b9th\s*Cir\.?|Ninth\s*Circuit", re.IGNORECASE)
RE_SPOUSE_CHILD = re.compile(
    r"(spouse|unmarried\s+child(?:ren)?)\s.*?(?:under\s+21|\bage\s*21\b|21\s*years)",
    re.IGNORECASE | re.DOTALL,
)
RE_KB_GAP = re.compile(r"knowledge\s*[- ]?base\s*gap", re.IGNORECASE)

# Generic legal citation extractor used by the no-hallucination check.
RE_ANY_CITE = re.compile(
    r"""
    (?P<cite>
      \b\d+\s*U\.?\s*S\.?\s*C\.?\s*§?\s*\d+[A-Za-z0-9\.\(\)]* |
      \b\d+\s*C\.?\s*F\.?\s*R\.?\s*§?\s*\d+[A-Za-z0-9\.\(\)]* |
      \bINA\s*§?\s*\d+[A-Za-z0-9\.\(\)]*
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _normalise_cite(s: str) -> str:
    """Reduce a citation to a comparable canonical form.

    Strips every non-alphanumeric character (whitespace, dots, ``§``, etc.)
    and lower-cases. This way ``"8 C.F.R. § 208.33"``, ``"8 CFR 208.33"``
    and ``"8 c f r § 208.33"`` all collapse to the same token, which lets
    the corpus-grounding and KB-gap checks compare apples to apples.
    """
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _chunks_text(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in chunks or []:
        t = c.get("text") or ""
        meta = c.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("original_text"):
            t = meta["original_text"]
        parts.append(t)
    return "\n\n".join(parts)


def check_derivative_concept(answer: str, _chunks) -> Tuple[bool, str]:
    hit = bool(RE_DERIVATIVE.search(answer) or RE_FOLLOW_TO_JOIN.search(answer))
    return hit, "found 'derivative asylum' / 'follow-to-join'" if hit else "missing derivative concept"


def check_ina_208_b3(answer: str, _chunks) -> Tuple[bool, str]:
    if RE_INA_208_B3.search(answer) or RE_USC_1158_B3.search(answer):
        return True, "INA § 208(b)(3) or 8 U.S.C. § 1158(b)(3) cited"
    return False, "missing INA § 208(b)(3) / 8 U.S.C. § 1158(b)(3)"


def check_cfr_208_21(answer: str, _chunks) -> Tuple[bool, str]:
    hit = bool(RE_CFR_208_21.search(answer))
    return hit, "8 C.F.R. § 208.21 cited" if hit else "missing 8 C.F.R. § 208.21"


def check_clp_named(answer: str, _chunks) -> Tuple[bool, str]:
    if RE_CLP_FULL.search(answer) or RE_CLP_ACR.search(answer):
        return True, "CLP named"
    return False, "CLP not named"


def check_cfr_208_33(answer: str, _chunks) -> Tuple[bool, str]:
    hit = bool(RE_CFR_208_33.search(answer))
    return hit, "8 C.F.R. § 208.33 (CLP rule) cited" if hit else "missing 8 C.F.R. § 208.33"


_CONCLUSION_POS = re.compile(
    r"(?:derivative[s]?|beneficiar(?:y|ies)|spouse|child(?:ren)?)\b"
    r"(?:[\s\w,()'\u2019\-]{0,120}?)"  # allow a short noun phrase between subject and verb
    r"\b(?:can|may|are\s+(?:eligible|able)|will\s+(?:receive|qualify|be\s+granted)|"
    r"qualify\s+for|qualif(?:y|ies)|"
    r"do\s+not\s+(?:need\s+to\s+)?(?:meet|satisfy|overcome|independently))\b",
    re.IGNORECASE,
)
_CONCLUSION_NEG = re.compile(
    r"(?:derivative[s]?|beneficiar(?:y|ies)|spouse|child(?:ren)?)\b"
    r"(?:[\s\w,()'\u2019\-]{0,120}?)"
    r"\b(?:cannot|may\s+not|are\s+(?:not\s+eligible|barred|ineligible)|"
    r"do(?:es)?\s+not\s+qualify|are\s+excluded)\b",
    re.IGNORECASE,
)


def check_correct_conclusion(answer: str, _chunks) -> Tuple[bool, str]:
    pos = list(_CONCLUSION_POS.finditer(answer))
    neg = list(_CONCLUSION_NEG.finditer(answer))
    if pos and len(pos) >= len(neg):
        return True, f"conclusion positive ({len(pos)} positive vs {len(neg)} negative phrases)"
    return False, f"conclusion negative or absent ({len(pos)} pos / {len(neg)} neg)"


_PRINC_DERIV = re.compile(
    r"(principal\s+applicant|main\s+applicant).*?(derivative|spouse|child)|(derivative|spouse|child).*?(principal\s+applicant|main\s+applicant)",
    re.IGNORECASE | re.DOTALL,
)


def check_principal_vs_derivative(answer: str, _chunks) -> Tuple[bool, str]:
    hit = bool(_PRINC_DERIV.search(answer))
    return hit, "draws principal-vs-derivative distinction" if hit else "no principal/derivative comparison"


def check_qualifying_relatives(answer: str, _chunks) -> Tuple[bool, str]:
    if RE_SPOUSE_CHILD.search(answer):
        return True, "mentions spouse / unmarried child under 21"
    if RE_INA_208_B3.search(answer) or RE_USC_1158_B3.search(answer):
        return True, "implicit via INA § 208(b)(3)(A) citation"
    return False, "no qualifying-derivative scope mentioned"


def check_inline_citations(answer: str, _chunks) -> Tuple[bool, str]:
    matches = RE_INLINE_CITE.findall(answer)
    if len(matches) >= 2:
        return True, f"{len(matches)} inline [S#] citations"
    return False, f"only {len(matches)} inline [S#] citations"


def check_no_hallucinated_citations(answer: str, chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    corpus = _normalise_cite(_chunks_text(chunks))
    found = list({m.group("cite") for m in RE_ANY_CITE.finditer(answer)})
    if not found:
        return True, "no citations in answer (vacuously grounded)"
    missing = [c for c in found if _normalise_cite(c) not in corpus]
    if not missing:
        return True, f"all {len(found)} citations grounded in retrieved chunks"
    sample = ", ".join(missing[:3])
    return False, f"{len(missing)} citation(s) not in retrieved chunks (e.g. {sample})"


def check_ninth_circuit_hook(answer: str, chunks) -> Tuple[bool, str]:
    has_corpus_hook = bool(RE_NINTH_CIR.search(_chunks_text(chunks)))
    answer_hit = bool(RE_NINTH_CIR.search(answer))
    if not has_corpus_hook:
        return True, "no 9th-Cir material in retrieved chunks — check skipped"
    return answer_hit, "answer references 9th Cir." if answer_hit else "missed 9th-Cir hook present in corpus"


def check_kb_gap_section(answer: str, chunks) -> Tuple[bool, str]:
    corpus = _normalise_cite(_chunks_text(chunks))
    # Each authority is satisfied if ANY of the listed canonical forms appears
    # in the corpus (e.g., USCIS form § 208.33 or EOIR form § 1208.33).
    expected = [
        ("8 C.F.R. § 208.33", ["8cfr20833", "8cfr120833"]),
        ("8 C.F.R. § 208.21", ["8cfr20821", "8cfr120821"]),
        ("INA § 208(b)(3)", ["ina208b3", "8usc1158b3"]),
    ]
    missing_from_corpus = [
        label for label, keys in expected if not any(k in corpus for k in keys)
    ]
    has_gap_section = bool(RE_KB_GAP.search(answer))
    if not missing_from_corpus:
        return True, "no gaps to flag — check passes vacuously"
    if has_gap_section:
        return True, f"KB Gap section present; corpus missing: {missing_from_corpus}"
    return False, f"corpus missing {missing_from_corpus} but no KB Gap section in answer"


# ---------------------------------------------------------------------------
# Rubric assembly
# ---------------------------------------------------------------------------
CheckFn = Callable[[str, List[Dict[str, Any]]], Tuple[bool, str]]

RUBRIC: List[Tuple[str, CheckFn, float]] = [
    ("derivative_concept", check_derivative_concept, 1.0),
    ("cite_ina_208_b3", check_ina_208_b3, 1.0),
    ("cite_cfr_208_21", check_cfr_208_21, 1.0),
    ("clp_named", check_clp_named, 1.0),
    ("cite_cfr_208_33", check_cfr_208_33, 1.0),
    ("correct_conclusion", check_correct_conclusion, 2.0),  # weighted higher
    ("principal_vs_derivative", check_principal_vs_derivative, 1.0),
    ("qualifying_relatives", check_qualifying_relatives, 0.5),
    ("inline_citations", check_inline_citations, 1.0),
    ("no_hallucinated_citations", check_no_hallucinated_citations, 2.0),
    ("ninth_circuit_hook", check_ninth_circuit_hook, 0.5),
    ("kb_gap_section", check_kb_gap_section, 0.5),
]


def grade(answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    results: List[CheckResult] = []
    for name, fn, weight in RUBRIC:
        try:
            passed, detail = fn(answer or "", retrieved or [])
        except Exception as e:  # pragma: no cover - rubric should not crash run
            passed, detail = False, f"check exception: {e}"
        results.append(CheckResult(name=name, passed=bool(passed), weight=weight, detail=detail))

    total_weight = sum(r.weight for r in results)
    earned = sum(r.weight for r in results if r.passed)
    return {
        "passed": [r.name for r in results if r.passed],
        "failed": [r.name for r in results if not r.passed],
        "score": round(earned / total_weight, 3) if total_weight else 0.0,
        "earned": earned,
        "max": total_weight,
        "details": [
            {"name": r.name, "passed": r.passed, "weight": r.weight, "detail": r.detail}
            for r in results
        ],
    }
