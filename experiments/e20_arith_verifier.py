"""
E20 — Deterministic arithmetic verifier as a pool selector (ZERO-GPU, numeric benches).

E19: gold-free VOTE-counting over the preserved pool beats final-majority but captures only ~12%
of the ceiling — the value is in a real VERIFIER, not vote-counting. For numeric tasks the cheapest
real verifier is DETERMINISTIC and needs no GPU: check whether a candidate answer is arithmetically
GROUNDED — i.e., it is the result of a numerically-correct equation inside its own supporting
reasoning. Select the grounded candidate. This targets self-error DRIFT (E4a's dominant GSM8K
mechanism): an agent that changed its answer without the arithmetic backing it up is penalized.

Selectors (all gold-free; gold only labels correctness afterward):
  final_majority   deployed baseline (mode of final-round buckets)
  hist_frequency   best simple E19 selector (most round×agent votes)
  arith_verify     pool candidate that is arithmetically grounded (answer = RHS of a VALID eq in
                   some supporting reasoning); rank (grounded, #valid-eqs-yielding-it, frequency)
  arith_x_freq     frequency, but grounded candidates strictly preferred (verifier as re-ranker)
Ceiling: oracle_history (perfect gold-free selector over the pool).

Reports verifier COVERAGE (fraction of debates where ≥1 candidate is grounded) — honesty about how
often the deterministic check even fires. If arith_verify beats frequency significantly, a real
verifier closes the E18 gap and the LLM-verifier (GPU) is well-justified for reasoning benches; if
not, deterministic arithmetic is insufficient -> the selector must be semantic (GPU judge).
Cluster-boot CI on qid. ZERO-GPU. Run:
  CUDA_VISIBLE_DEVICES="" python experiments/e20_arith_verifier.py
"""
import os, sys, glob, json, re, ast, operator
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from src.utils.io_utils import read_jsonl
import experiments.e3_router as e3

BENCHES = ["gsm8k", "math"]           # numeric-answer benches only
OUT = "results/tables/e20_arith_verifier.json"
B_BOOT = 2000
SELECTORS = ["final_majority", "hist_frequency", "arith_verify", "arith_x_freq"]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}
_EXPR = re.compile(r"\d[\d\s,\.\+\-\*/×÷x\(\)]*[\d\)]")   # arithmetic-looking substrings


def _f(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError


def valid_results(reasoning):
    """Numeric values of arithmetic EXPRESSIONS embedded in the reasoning (e.g. '(47+52+57)/3'),
    safely evaluated. These are the answer values the reasoning actually COMPUTES."""
    out = []
    for m in _EXPR.findall(reasoning or ""):
        s = m.replace("×", "*").replace("÷", "/").replace("x", "*").replace(",", "").strip()
        if not any(o in s for o in "+-*/") or not any(c.isdigit() for c in s):
            continue
        try:
            v = _safe_eval(ast.parse(s, mode="eval"))
        except Exception:
            continue
        if isinstance(v, (int, float)) and abs(v) < 1e12:
            out.append(float(v))
    return out


def _pick(scores, freq):
    if not scores:
        return None
    return max(scores, key=lambda b: (scores[b], freq[b], b))


def debate_record(lg, bench):
    p = e3.prep_log(lg, bench)
    F, agents = p["F"], p["agents"]
    cells = [(r, a, p["B"][r][a]) for r in range(F + 1) for a in agents if p["B"][r][a] is not None]
    if not cells:
        return None
    pool = {b for _, _, b in cells}
    correct = {b for r in range(F + 1) for a in agents
               if p["B"][r][a] is not None and p["C"][r][a] for b in [p["B"][r][a]]}
    freq = Counter(b for _, _, b in cells)
    final_b = Counter(p["B"][F][a] for a in agents if p["B"][F][a] is not None)

    # arithmetic grounding: does a valid equation in a supporting reasoning yield the candidate value?
    ground = {b: 0 for b in pool}
    for r, a, b in cells:
        val = _f(b)
        if val is None:
            continue
        for z in valid_results(p["R"][r][a]):
            if abs(z - val) <= 1e-6 + 1e-3 * abs(val):
                ground[b] += 1
    grounded_any = any(v > 0 for v in ground.values())

    scores = {
        "final_majority": dict(final_b),
        "hist_frequency": dict(freq),
        "arith_verify": {b: (1 if ground[b] > 0 else 0) * 1000 + ground[b] for b in pool},
        "arith_x_freq": {b: (1 if ground[b] > 0 else 0) * 100000 + freq[b] for b in pool},
    }
    rec = {name: (_pick(sc, freq) in correct) for name, sc in scores.items()}
    rec["oracle_history"] = bool(pool & correct)
    rec["grounded_any"] = grounded_any
    return rec


def rows_for(bench):
    rows = []
    for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
        for lg in read_jsonl(f):
            r = debate_record(lg, bench)
            if r is not None:
                r["qid"] = lg["question_id"]
                rows.append(r)
    return rows


def _mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


def boot_delta(rows, key, ref, B=B_BOOT):
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["qid"]].append(r)
    cl = list(clusters.values())
    rng = np.random.default_rng(0)
    d = np.empty(B)
    for b in range(B):
        samp = [x for j in rng.integers(0, len(cl), len(cl)) for x in cl[j]]
        d[b] = np.mean([x[key] for x in samp]) - np.mean([x[ref] for x in samp])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summarize(rows, label):
    fm = _mean(rows, "final_majority")
    oh = _mean(rows, "oracle_history")
    cov = _mean(rows, "grounded_any")
    print(f"\n=== {label}  (n={len(rows)}, {len({r['qid'] for r in rows})} qid) ===")
    print(f"  verifier coverage (>=1 grounded candidate): {cov:.3f}")
    print(f"  {'selector':<16}{'acc':>7}{'Δ vs final-maj':>20}{'Δ vs hist-freq':>20}")
    res = {"grounded_coverage": cov}
    for s in SELECTORS:
        acc = _mean(rows, s)
        line = f"  {s:<16}{acc:>7.3f}"
        entry = {"acc": acc}
        if s != "final_majority":
            lo, hi = boot_delta(rows, s, "final_majority")
            sig = "*" if lo > 0 else " "
            line += f"   {acc - fm:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}"
            entry.update({"d_final": acc - fm, "ci_final": [lo, hi], "sig_final": lo > 0})
        else:
            line += f"{'(baseline)':>20}"
        if s in ("arith_verify", "arith_x_freq"):
            lo2, hi2 = boot_delta(rows, s, "hist_frequency")
            sig2 = "*" if lo2 > 0 else " "
            line += f"   {acc - _mean(rows,'hist_frequency'):+.3f} [{lo2:+.3f},{hi2:+.3f}]{sig2}"
            entry.update({"d_freq": acc - _mean(rows, "hist_frequency"),
                          "ci_freq": [lo2, hi2], "sig_freq": lo2 > 0})
        print(line)
        res[s] = entry
    print(f"  {'oracle_history':<16}{oh:>7.3f}   {oh - fm:+.3f}  (ceiling)")
    res["oracle_history"] = oh
    return res


def main():
    out, allrows = {}, []
    for bench in BENCHES:
        rows = rows_for(bench)
        if not rows:
            print(f"  [{bench}] no logs, skip"); continue
        out[bench] = summarize(rows, bench)
        allrows += rows
    if len({r["qid"] for r in allrows}) and len(out) > 1:
        out["pooled"] = summarize(allrows, "POOLED (numeric)")
    out["_note"] = ("arith_verify = deterministic; picks a candidate whose value is the result of a "
                    "numerically-VALID equation in its supporting reasoning. Δ vs hist-freq is the "
                    "decisive test (does verification add over vote-counting?). Coverage = how often "
                    "the check fires. Low/insignificant gain => deterministic arithmetic insufficient, "
                    "the selector must be a semantic LLM judge (GPU).")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
