"""
E22 — Does PROVENANCE distinguish correct vs incorrect hypotheses? (ZERO-GPU diagnostic).

Motivation: E21 refuted the ARCHITECTURE "weak 3B debate -> strong EXTERNAL judge" (a capable judge
does better solving than selecting from a 3B pool), NOT preservation. E18/E19 stand: preservation
adds +12.5pp candidate recall and cheap gold-free heuristics already add ~+3pp WITHOUT any extra
model. The natural HP-MAD selector therefore scores each hypothesis from ITS OWN HISTORY — no new
inference, no capability mismatch:
    S(h) = w1·N_independent(h) + w2·N_persistent(h) + w3·N_support(h) - w4·N_late_adoption(h)  ...

BUT do NOT tune a selector first. First ask, per signal, candidate-level:
    P(correct | N_independent),  P(correct | persistence),  P(correct | late_adoption),  ...
If independent-support / persistence rise MONOTONICALLY with correctness ACROSS benchmarks (and
late-adoption / resurrection fall) -> GO build a frozen provenance selector. If flat -> real STOP.

Candidate = one distinct answer-bucket in a debate's history pool (rounds 0..F). Same extraction /
correctness as every prior experiment (e3.prep_log). "Independent" = round 0 (agents' first answer,
pre-revision; matches the independent_answers field). Correlations within a debate are handled by
clustering the bootstrap on qid. ZERO-GPU. Run:
  CUDA_VISIBLE_DEVICES="" python experiments/e22_provenance.py
"""
import os, sys, glob, json
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.io_utils import read_jsonl
import experiments.e3_router as e3

BENCHES = ["gsm8k", "bbh", "strategyqa", "commonsenseqa", "truthfulqa"]
OUT = "results/tables/e22_provenance.json"
B_BOOT = 2000

# each signal: (name, expected sign vs correctness, bin edges as list of (label, predicate))
SIGNALS = ["n_independent", "n_persist", "n_support", "first_round", "n_late_adopt", "resurrected"]
EXPECT = {"n_independent": +1, "n_persist": +1, "n_support": +1,
          "first_round": -1, "n_late_adopt": -1, "resurrected": -1}


def candidates_for(bench, logdir="data/debate_logs"):
    """One row per distinct hypothesis-bucket per debate, with provenance features + correct label.
    logdir overridable so E25 reuses this exact candidate builder on alt-ensemble logs."""
    rows = []
    for f in sorted(glob.glob(f"{logdir}/{bench}_seed*.jsonl")):
        for lg in read_jsonl(f):
            p = e3.prep_log(lg, bench)
            F, agents = p["F"], p["agents"]
            correct = {b for r in range(F + 1) for a in agents
                       if p["B"][r][a] is not None and p["C"][r][a] for b in [p["B"][r][a]]}
            cells = [(r, a) for r in range(F + 1) for a in agents if p["B"][r][a] is not None]
            pool = {p["B"][r][a] for r, a in cells}
            for h in pool:
                present_rounds = sorted({r for r, a in cells if p["B"][r][a] == h})
                # late adoption: agent switches TO h at round>=1 (didn't hold it the round before)
                late = sum(1 for r in range(1, F + 1) for a in agents
                           if p["B"][r][a] == h and p["B"][r - 1][a] != h)
                span = set(range(present_rounds[0], present_rounds[-1] + 1))
                resurrected = int(len(span) > len(present_rounds))     # a gap -> disappeared & returned
                rows.append({
                    "bench": bench, "qid": lg["question_id"], "correct": int(h in correct),
                    "n_independent": sum(1 for a in agents if p["B"][0][a] == h),   # round-0 support
                    "n_support": sum(1 for r, a in cells if p["B"][r][a] == h),
                    "n_persist": len(present_rounds),
                    "first_round": present_rounds[0],
                    "n_late_adopt": late,
                    "resurrected": resurrected,
                })
    return rows


def pcorrect_by_bin(rows, sig):
    """P(correct | signal value) for each distinct integer value, with n."""
    by = defaultdict(list)
    for r in rows:
        by[r[sig]].append(r["correct"])
    return {int(v): (float(np.mean(c)), len(c)) for v, c in sorted(by.items())}


def signal_auc(rows, sig):
    y = np.array([r["correct"] for r in rows])
    x = np.array([r[sig] for r in rows], float)
    if len(set(y)) < 2 or len(set(x)) < 2:
        return float("nan"), (float("nan"), float("nan"))
    auc = roc_auc_score(y, x)
    cl = defaultdict(list)
    for i, r in enumerate(rows):
        cl[r["qid"]].append(i)
    cl = list(cl.values())
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(B_BOOT):
        idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
        yy, xx = y[idx], x[idx]
        if len(set(yy)) == 2 and len(set(xx)) > 1:
            boot.append(roc_auc_score(yy, xx))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan")))
    return float(auc), (float(lo), float(hi))


def monotone(bins, sign):
    """Is P(correct|value) monotone in the expected direction over values with n>=20?"""
    pts = [(v, m) for v, (m, n) in bins.items() if n >= 20]
    if len(pts) < 2:
        return None
    ys = [m for _, m in sorted(pts)]
    diffs = np.diff(ys)
    return bool(np.all(diffs >= -1e-9)) if sign > 0 else bool(np.all(diffs <= 1e-9))


def summarize(rows, label, per_bench_rows=None):
    base = float(np.mean([r["correct"] for r in rows]))
    print(f"\n=== {label}  (candidates={len(rows)}, base P(correct)={base:.3f}) ===")
    res = {"n_candidates": len(rows), "base_rate": base, "signals": {}}
    for sig in SIGNALS:
        auc, (lo, hi) = signal_auc(rows, sig)
        bins = pcorrect_by_bin(rows, sig)
        exp = EXPECT[sig]
        # directional AUC: >0.5 means signal predicts correct in EXPECTED direction
        dir_auc = auc if exp > 0 else (1 - auc)
        dlo, dhi = (lo, hi) if exp > 0 else (1 - hi, 1 - lo)
        mono_pool = monotone(bins, exp)
        n_agree = None
        if per_bench_rows:
            agree = [monotone(pcorrect_by_bin(br, sig), exp) for br in per_bench_rows.values()]
            n_agree = sum(1 for a in agree if a is True)
        sig_flag = "*" if (not np.isnan(dlo) and dlo > 0.5) else " "
        binstr = " ".join(f"{v}:{m:.2f}(n{n})" for v, (m, n) in bins.items() if n >= 20)
        agreestr = f" | mono {n_agree}/5 bench" if n_agree is not None else ""
        print(f"  {sig:<14} dirAUC {dir_auc:.3f} [{dlo:.3f},{dhi:.3f}]{sig_flag}"
              f"  mono={mono_pool}{agreestr}")
        print(f"      P(correct|·): {binstr}")
        res["signals"][sig] = {"auc": auc, "dir_auc": dir_auc, "dir_ci": [dlo, dhi],
                               "expected_sign": exp, "monotone_pooled": mono_pool,
                               "monotone_benches": n_agree, "bins": {str(k): v for k, v in bins.items()}}
    # untuned equal-weight combo (PREVIEW ONLY, not a tuned selector): z-sum with expected signs
    X = {s: np.array([r[s] for r in rows], float) for s in SIGNALS}
    z = np.zeros(len(rows))
    for s in SIGNALS:
        col = X[s]
        sd = col.std()
        if sd > 0:
            z += EXPECT[s] * (col - col.mean()) / sd
    y = np.array([r["correct"] for r in rows])
    combo_auc = float(roc_auc_score(y, z)) if len(set(y)) == 2 else float("nan")
    print(f"  {'[combo]':<14} untuned equal-weight z-sum AUC = {combo_auc:.3f}  (preview, NOT tuned)")
    res["untuned_combo_auc"] = combo_auc
    return res


def main():
    out, allrows, per_bench = {}, [], {}
    for bench in BENCHES:
        rows = candidates_for(bench)
        if rows:
            per_bench[bench] = rows
            allrows += rows
    for bench, rows in per_bench.items():
        out[bench] = summarize(rows, bench)
    out["pooled"] = summarize(allrows, "POOLED", per_bench_rows=per_bench)
    out["_note"] = ("Candidate-level P(correct|provenance). dirAUC>0.5 (*, CI-lo>0.5) => the signal "
                    "predicts correctness in its EXPECTED direction. GO for a frozen provenance "
                    "selector iff independent-support / persistence are significant AND monotone "
                    "across most benchmarks. Combo AUC is an UNtuned equal-weight preview only.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
