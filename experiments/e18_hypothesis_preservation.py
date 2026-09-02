"""
E18 — Hypothesis-Preservation offline ceiling (ZERO-GPU GO/NO-GO).

MOTIVATION: E13-E17 show that TARGETING intervention (predict who benefits) is the hard,
possibly-impossible part (five preregistered negatives). So stop predicting benefit; instead
attack the structural flaw the data already proves — ANSWER EXTINCTION — with an intervention
that needs no targeting: Non-Destructive / Hypothesis-Preserving debate (HP-MAD). Debate may
ADD hypotheses but never DESTRUCTIVELY overwrite a previously-supported one; the final answer is
selected from the whole history pool H_T = ∪_{r≤T} {distinct answers at round r}, not just the
final round. Monotonic candidate set: H_t ⊆ H_{t+1}.

THIS SCRIPT = the cheapest possible test, BEFORE any GPU or any selector. Using existing debate
logs and the SAME answer-extraction+correctness as every prior experiment (e3.prep_log →
robust_answer / is_correct / e4a.bucket), it measures the ORACLE ceiling that preservation buys:

  CR_final   = P(y* is present among FINAL-round candidate answers)        [oracle | current MAD]
  CR_history = P(y* is present ANYWHERE across rounds 0..T)                [oracle | HP-MAD pool]
  ΔCR        = CR_history − CR_final = P(correct appeared but was FORGOTTEN by the final round)

ΔCR is the structural value of preservation: it is exactly the accuracy a PERFECT selector over
the preserved pool would recover over a perfect selector over the final round. If ΔCR is large
(user gate: ≳ +5-10 pp), the story is "debate does not LACK the right answer — it FORGETS it,"
and building a verifier/selector over the pool has real headroom → GO. If ΔCR ≈ 0 → the correct
answer is not being lost, preservation buys nothing → STOP.

Also reports current majority accuracy (mode of final buckets) as the deployed baseline, and pool
size/cost (distinct hypotheses to adjudicate). Cluster-bootstrap CI on ΔCR over qid (seeds of a
question are correlated repeats). ZERO-GPU. Run:
  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e18_hypothesis_preservation.py
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
OUT = "results/tables/e18_hypothesis_preservation.json"
B_BOOT = 2000


def rows_for(bench, logdir="data/debate_logs"):
    """One row per (qid, seed) debate: recall + pool-size descriptors, gold via e3.prep_log.
    logdir overridable so E25 can point the SAME metric at alt-ensemble logs (default unchanged)."""
    rows = []
    for f in sorted(glob.glob(f"{logdir}/{bench}_seed*.jsonl")):
        seed = int(f.split("_seed")[1].split(".")[0])
        for lg in read_jsonl(f):
            p = e3.prep_log(lg, bench)
            F, agents = p["F"], p["agents"]
            appeared = any(p["ncorr"][r] >= 1 for r in range(F + 1))   # correct seen anywhere
            final = p["ncorr"][F] >= 1                                  # correct at final round
            hist_pool = {p["B"][r][a] for r in range(F + 1) for a in agents if p["B"][r][a] is not None}
            final_pool = {p["B"][F][a] for a in agents if p["B"][F][a] is not None}
            fb = [p["B"][F][a] for a in agents if p["B"][F][a] is not None]      # majority (mode) vote
            if fb:
                win = Counter(fb).most_common(1)[0][0]
                maj = any(p["C"][F][a] for a in agents if p["B"][F][a] == win)
            else:
                maj = False
            rows.append({"qid": lg["question_id"], "seed": seed,
                         "appeared": appeared, "final": final, "extinct": appeared and not final,
                         "maj": maj, "hist_size": len(hist_pool), "final_size": len(final_pool)})
    return rows


def _dcr(rows):
    ap = np.mean([r["appeared"] for r in rows])
    fi = np.mean([r["final"] for r in rows])
    return ap - fi


def boot_ci_dcr(rows, B=B_BOOT):
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["qid"]].append(r)
    cl = list(clusters.values())
    rng = np.random.default_rng(0)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(cl), len(cl))
        samp = [x for j in idx for x in cl[j]]
        vals[b] = _dcr(samp)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(rows, label):
    n = len(rows)
    cr_hist = float(np.mean([r["appeared"] for r in rows]))
    cr_final = float(np.mean([r["final"] for r in rows]))
    maj = float(np.mean([r["maj"] for r in rows]))
    dcr = cr_hist - cr_final
    lo, hi = boot_ci_dcr(rows)
    pool = float(np.mean([r["hist_size"] for r in rows]))
    fpool = float(np.mean([r["final_size"] for r in rows]))
    gate = "GO" if lo > 0 else "no"
    print(f"\n=== {label}  (n={n}, {len({r['qid'] for r in rows})} qid) ===")
    print(f"  current maj-acc (mode)          {maj:.3f}")
    print(f"  CR_final   (oracle | final)     {cr_final:.3f}")
    print(f"  CR_history (oracle | HP pool)   {cr_hist:.3f}")
    print(f"  ΔCR = forgotten-correct         {dcr:+.3f}  [{lo:+.3f},{hi:+.3f}]  {gate}")
    print(f"  extinction rate                 {np.mean([r['extinct'] for r in rows]):.3f}")
    print(f"  pool size hist / final          {pool:.2f} / {fpool:.2f}  (+{pool - fpool:.2f} candidates)")
    return {"n": n, "n_qid": len({r["qid"] for r in rows}),
            "maj_acc_mode": maj, "CR_final": cr_final, "CR_history": cr_hist,
            "dCR": dcr, "dCR_ci": [lo, hi], "gate_lo_gt_0": lo > 0,
            "extinction_rate": float(np.mean([r["extinct"] for r in rows])),
            "pool_hist": pool, "pool_final": fpool}


def main():
    out, allrows = {}, []
    for bench in BENCHES:
        rows = rows_for(bench)
        if not rows:
            print(f"  [{bench}] no logs, skip"); continue
        out[bench] = summarize(rows, bench)
        allrows += rows
    out["pooled"] = summarize(allrows, "POOLED")
    out["_note"] = ("ΔCR = P(correct appeared across rounds) - P(correct in final round) = the "
                    "oracle accuracy a perfect selector over the preserved pool recovers over the "
                    "final round. GO iff ΔCR CI lo > 0 (and practically ≳ +0.05). Selector/verifier "
                    "and GPU come AFTER a GO; this only tests the preservation ceiling.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
