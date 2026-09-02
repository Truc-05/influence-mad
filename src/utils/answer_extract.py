"""Robust, deterministic answer extraction for JOURNAL re-scoring.

The conference logging stored a `{answer, reasoning}` pair per agent per round,
where `answer` came from a regex on the model's `ANSWER:` line. When generation
was truncated at max_tokens BEFORE the `ANSWER:` line, `answer` degenerates to
"REASONING:" and the true answer is either (a) recoverable from a structured cue
in the reasoning (e.g. \\boxed{...}) or (b) genuinely absent (the model never
concluded). This module recovers (a) and flags (b) as CENSORED so that
truncation artifacts do not inflate behavioral extinction / headroom.

Design: conservative. We only look past the `answer` field when it is degenerate,
and we only accept STRUCTURED cues (boxed / explicit answer keyword) -- never a
bare "last number in the reasoning", which would fabricate answers from truncated
arithmetic. Closed-answer benchmarks (yes/no, multiple-choice) had ~0 truncation
failures, so their path is a near-passthrough.

robust_answer(answer, reasoning, bench) -> str | None      (None == censored)
"""
import re
from src.utils.scoring import _extract_number, _normalize_text

_DEGENERATE = {"", "reasoning", "reasoning:", "answer", "answer:"}


def _degenerate(answer):
    a = (answer or "").strip().lower()
    return a in _DEGENERATE or (a.startswith("reasoning:") and _extract_number(a) is None)


def _boxed(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    return m[-1] if m else None


def _after_answer_kw(text):
    # "final answer is 42", "the answer: 42", "answer = 42"
    m = re.findall(r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*([^\n.]{1,40})", text or "", re.I)
    return m[-1] if m else None


def robust_answer(answer, reasoning, bench):
    """Return a best-effort answer STRING to feed is_correct, or None if no
    reliable answer can be recovered (censored)."""
    a = (answer or "").strip()

    if bench in ("gsm8k", "svamp"):
        if not _degenerate(answer) and _extract_number(a) is not None:
            return a
        for cand in (_boxed(reasoning), _boxed(a), _after_answer_kw(reasoning)):
            if cand and _extract_number(cand) is not None:
                return cand
        return None                                  
    
    if not _degenerate(answer) and a:
        return a
    # rare degenerate case: try a structured cue, else censor
    for cand in (_boxed(reasoning), _after_answer_kw(reasoning)):
        if cand and _normalize_text(cand):
            return cand
    return None
