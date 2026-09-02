"""
E19 — Gold-free selector bake-off over the preserved hypothesis pool (ZERO-GPU).

E18 proved a large ORACLE ceiling for hypothesis preservation (pooled CR_history 0.760 vs
final-majority 0.509, ΔCR +0.125). That ceiling assumes a PERFECT gold-free selector. This asks
the next cheap question BEFORE any GPU/LLM-verifier: can a SIMPLE gold-free heuristic picking one
answer from the whole-trajectory pool already beat the deployed final-majority, and how much of
the ceiling does it capture? If yes, a real selector is plausible and the story is de-risked.

Every selector is GOLD-FREE (uses only the transcript: which answer, which round, which agent —
never ground truth). Same extraction/correctness as all prior exp (e3.prep_log). A pick is
correct iff its answer-bucket equals a bucket that IS correct for this question.

Selectors:
  final_majority   mode of FINAL-round buckets (the DEPLOYED baseline; final view only)
  round0_majority  mode of ROUND-0 buckets (pre-debate independent consensus; does debate help?)
  hist_frequency   most (round×agent) votes across ALL rounds  (self-consistency-style)
  persistence      bucket alive in the MOST distinct rounds (survives longest); tie→frequency
  recency          appearances weighted by round (+later); tests if trusting late debate helps
  antirecency      appearances weighted (+earlier); tests if trusting early answers helps
Reference ceilings (not deployable selectors): oracle_final (=CR_final), oracle_history (=CR_history).

hist_frequency / persistence / recency / antirecency select from the WHOLE pool, so they CAN
resurrect a forgotten-correct answer (the HP-MAD advantage); the two majority selectors cannot.
Ties broken deterministically by (score, total-frequency, bucket-string). Cluster-boot CI on qid
for each selector's Δ vs final_majority (paired). ZERO-GPU. Run:
  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e19_selector_bakeoff.py
"""
import os, sys, glob, json
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from src.utils.io_utils import read_jsonl
import experiments.e3_router as e3

BENCHES = ["gsm8k", "bbh", "strategyqa", "commonsenseqa", "truthfulqa"]
OUT = "results/tables/e19_selector_bakeoff.json"
B_BOOT = 2000
SELECTORS = ["final_majority", "round0_majority", "hist_frequency",
             "persistence", "recency", "antirecency"]


def _pick(scores, freq):
    """argmax with deterministic tie-break: higher score, then higher total freq, then bucket str."""
    if not scores:
        return None
    return max(scores, key=lambda b: (scores[b], freq[b], b))


def debate_record(lg, bench):
    """All selector picks (correct/incorrect) + oracle flags for one debate. Gold-free picks;
    gold only labels correctness after the fact."""
    p = e3.prep_log(lg, bench)
    F, agents = p["F"], p["agents"]
    cells = [(r, a, p["B"][r][a]) for r in range(F + 1) for a in agents if p["B"][r][a] is not None]
    if not cells:
        return None
    pool = {b for _, _, b in cells}
    correct_buckets = {b for r in range(F + 1) for a in agents
                       if p["B"][r][a] is not None and p["C"][r][a] for b in [p["B"][r][a]]}
    freq = Counter(b for _, _, b in cells)
    rounds_of = defaultdict(set)
    for r, _, b in cells:
        rounds_of[b].add(r)
    final_b = Counter(p["B"][F][a] for a in agents if p["B"][F][a] is not None)
    round0_b = Counter(p["B"][0][a] for a in agents if p["B"][0][a] is not None)

    scores = {
        "final_majority": dict(final_b),
        "round0_majority": dict(round0_b),
        "hist_frequency": dict(freq),
        "persistence": {b: len(rs) for b, rs in rounds_of.items()},
        "recency": {b: sum(r + 1 for r, _, bb in cells if bb == b) for b in pool},
        "antirecency": {b: sum(F + 1 - r for r, _, bb in cells if bb == b) for b in pool},
    }
    rec = {name: (_pick(sc, freq) in correct_buckets) for name, sc in scores.items()}
    rec["oracle_final"] = any(p["B"][F][a] in correct_buckets for a in agents if p["B"][F][a] is not None)
    rec["oracle_history"] = bool(pool & correct_buckets)
    rec["pool"] = len(pool)
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


def boot_delta(rows, key, ref="final_majority", B=B_BOOT):
    """Paired cluster-boot CI of mean(key) - mean(ref) over qid clusters."""
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
    of, oh = _mean(rows, "oracle_final"), _mean(rows, "oracle_history")
    head = oh - fm                                      # total headroom over deployed baseline
    print(f"\n=== {label}  (n={len(rows)}, {len({r['qid'] for r in rows})} qid) ===")
    print(f"  {'selector':<16}{'acc':>7}{'Δ vs final-maj':>20}{'%headroom':>11}")
    res = {}
    for s in SELECTORS:
        acc = _mean(rows, s)
        if s == "final_majority":
            print(f"  {s:<16}{acc:>7.3f}{'(baseline)':>20}")
            res[s] = {"acc": acc}
            continue
        lo, hi = boot_delta(rows, s)
        sig = "*" if lo > 0 else (" " if hi > 0 else "-")
        pct = (acc - fm) / head * 100 if head > 0 else float("nan")
        print(f"  {s:<16}{acc:>7.3f}   {acc - fm:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}{pct:>9.0f}%")
        res[s] = {"acc": acc, "delta": acc - fm, "ci": [lo, hi], "sig": lo > 0, "pct_headroom": pct}
    print(f"  {'oracle_final':<16}{of:>7.3f}   {of - fm:+.3f}  (ceiling: best FINAL answer)")
    print(f"  {'oracle_history':<16}{oh:>7.3f}   {oh - fm:+.3f}  (ceiling: HP pool)")
    res.update({"oracle_final": of, "oracle_history": oh, "headroom": head,
                "pool_mean": _mean(rows, "pool")})
    return res


def main():
    out, allrows = {}, []
    for bench in BENCHES:
        rows = rows_for(bench)
        if not rows:
            continue
        out[bench] = summarize(rows, bench)
        allrows += rows
    out["pooled"] = summarize(allrows, "POOLED")
    out["_note"] = ("All selectors gold-free. Δ vs final_majority with qid cluster-boot 95% CI; "
                    "* = lo>0 (beats deployed baseline). %headroom = fraction of "
                    "(oracle_history - final_majority) captured. A significant positive pool "
                    "selector de-risks HP-MAD before the GPU LLM-verifier run.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
