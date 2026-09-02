"""
E24 — Convergence-gated provenance selector: beat the round-0 baseline (ZERO-GPU).

E23's provenance selector ties round0_majority because it essentially trusts EARLY support and
ignores the final round — so it LOSES on StrategyQA, where debate is often CONSTRUCTIVE (legitimate
later revision, not social copying). To beat round-0 we need a principled signal that keeps
independent evidence but CREDITS legitimate later consensus while DISCOUNTING surface copying.

Signal (grounded in E3b-v3's finding that 73% of adoptions are SURFACE — answer matches a peer but
reasoning does NOT converge): reward final consensus only when the supporters' REASONING genuinely
converges, measured by lexical jaccard RESIDUALIZED against the debate's own baseline similarity
(topic-controlled, per E3b). No new LLM calls; no embeddings.

Features per candidate h (all gold-free, from e3.prep_log; reasoning = p["R"]):
  n_independent, n_persist, n_support, first_round      (E23 provenance)
  final_support   = agents holding h at round F
  conv_resid      = mean pairwise reasoning-jaccard among h's supporters − debate baseline jaccard
  final_stability = fraction of round-F supporters of h who ALSO held it at round F-1

Selectors (gold-free, same e3.prep_log correctness):
  final_majority          deployed baseline
  round0_majority         **the baseline to beat** (E23's strongest simple selector)
  prov_logistic_xfit      E23 provenance (reference)
  convgate_hand           interpretable: z(indep)+z(persist)−z(first_round)+z(final_support·max(conv,0))
  provconv_logistic_xfit  ALL features, LogisticRegression cross-fit by qid (OOF; never on test)

PRIMARY: ΔAcc vs **round0_majority** (and vs final_majority), cluster-boot qid, per-bench. Success =
beat round0 pooled (CI>0) AND fix StrategyQA (Δ vs final_maj no longer negative). ZERO-GPU. Run:
  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e24_convergence_selector.py
"""
import os, sys, glob, json
from collections import Counter, defaultdict
from itertools import combinations

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.utils.io_utils import read_jsonl
import experiments.e3_router as e3

BENCHES = ["gsm8k", "bbh", "strategyqa", "commonsenseqa", "truthfulqa"]
OUT = "results/tables/e24_convergence_selector.json"
B_BOOT = 4000
ALLF = ["n_independent", "n_persist", "n_support", "first_round",
        "final_support", "conv_resid", "final_stability"]


def _jacc(a, b):
    sa, sb = set((a or "")[:500].lower().split()), set((b or "")[:500].lower().split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def _pick(scores, freq):
    if not scores:
        return None
    return max(scores, key=lambda b: (scores[b], freq[b], str(b)))


def build_debates():
    debates = []
    for bench in BENCHES:
        for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
            for lg in read_jsonl(f):
                p = e3.prep_log(lg, bench)
                F, agents = p["F"], p["agents"]
                cells = [(r, a) for r in range(F + 1) for a in agents if p["B"][r][a] is not None]
                if not cells:
                    continue
                correct = {b for r in range(F + 1) for a in agents
                           if p["B"][r][a] is not None and p["C"][r][a] for b in [p["B"][r][a]]}
                pool = {p["B"][r][a] for r, a in cells}
                freq = Counter(p["B"][r][a] for r, a in cells)
                # debate baseline reasoning similarity (topic control)
                reas = [p["R"][r][a] for r, a in cells]
                base = float(np.mean([_jacc(x, y) for x, y in combinations(reas, 2)])) if len(reas) > 1 else 0.0
                final_holders = [a for a in agents if p["B"][F][a] is not None]
                feats = {}
                for h in pool:
                    pr = sorted({r for r, a in cells if p["B"][r][a] == h})
                    supp_reas = [p["R"][r][a] for r, a in cells if p["B"][r][a] == h]
                    conv = (float(np.mean([_jacc(x, y) for x, y in combinations(supp_reas, 2)])) - base
                            if len(supp_reas) > 1 else 0.0)
                    fsupp = [a for a in final_holders if p["B"][F][a] == h]
                    stab = (np.mean([1.0 if (F >= 1 and p["B"][F - 1][a] == h) else 0.0 for a in fsupp])
                            if fsupp else 0.0)
                    feats[h] = {"n_independent": sum(1 for a in agents if p["B"][0][a] == h),
                                "n_support": freq[h], "n_persist": len(pr), "first_round": pr[0],
                                "final_support": len(fsupp), "conv_resid": conv,
                                "final_stability": float(stab)}
                final_b = Counter(p["B"][F][a] for a in agents if p["B"][F][a] is not None)
                round0_b = Counter(p["B"][0][a] for a in agents if p["B"][0][a] is not None)
                picks = {"final_majority": _pick(dict(final_b), freq),
                         "round0_majority": _pick(dict(round0_b), freq)}
                debates.append({"bench": bench, "qid": lg["question_id"], "pool": list(pool),
                                "feats": feats, "freq": freq, "correct": correct,
                                "CR_history": bool(pool & correct), "picks": picks})
    return debates


def add_learned(debates):
    flat = [(di, h) for di, d in enumerate(debates) for h in d["pool"]]
    Xall = np.array([[debates[di]["feats"][h][f] for f in ALLF] for di, h in flat], float)
    y = np.array([int(h in debates[di]["correct"]) for di, h in flat])
    groups = np.array([debates[di]["qid"] for di, h in flat])

    def argmax_per_debate(score, name):
        by = defaultdict(dict)
        for k, (di, h) in enumerate(flat):
            by[di][h] = score[k]
        for di, d in enumerate(debates):
            d["picks"][name] = _pick(by[di], d["freq"])

    # interpretable convergence-gated hand score (z-scored, gate consensus by positive convergence)
    def z(col):
        c = Xall[:, ALLF.index(col)].astype(float); s = c.std()
        return (c - c.mean()) / s if s > 0 else c * 0.0
    gate = Xall[:, ALLF.index("final_support")] * np.clip(Xall[:, ALLF.index("conv_resid")], 0, None)
    gate = (gate - gate.mean()) / (gate.std() if gate.std() > 0 else 1.0)
    hand = z("n_independent") + z("n_persist") - z("first_round") + gate
    argmax_per_debate(hand, "convgate_hand")

    # cross-fit logistic: E23 features, and the full convergence set
    ns = min(5, len(set(groups)))
    for name, cols in [("prov_logistic_xfit", ALLF[:4]),
                       ("provconv_logistic_xfit", ALLF)]:
        X = np.array([[debates[di]["feats"][h][f] for f in cols] for di, h in flat], float)
        oof = np.full(len(flat), np.nan)
        for tr, te in GroupKFold(ns).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        argmax_per_debate(oof, name)
    return ["prov_logistic_xfit", "convgate_hand", "provconv_logistic_xfit"]


def acc(debates, sel):
    return float(np.mean([d["picks"].get(sel) in d["correct"] for d in debates]))


def boot_delta(debates, sel, ref, B=B_BOOT):
    cl = defaultdict(list)
    for d in debates:
        cl[d["qid"]].append(d)
    cl = list(cl.values())
    rng = np.random.default_rng(0)
    out = np.empty(B)
    for i in range(B):
        s = [x for j in rng.integers(0, len(cl), len(cl)) for x in cl[j]]
        out[i] = (np.mean([d["picks"].get(sel) in d["correct"] for d in s])
                  - np.mean([d["picks"].get(ref) in d["correct"] for d in s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def summarize(debates, label, sels):
    fm, r0 = acc(debates, "final_majority"), acc(debates, "round0_majority")
    crh = float(np.mean([d["CR_history"] for d in debates]))
    print(f"\n=== {label}  (n={len(debates)}, final-maj {fm:.3f}, round0 {r0:.3f}, CR_h {crh:.3f}) ===")
    res = {"final_majority": fm, "round0_majority": r0, "CR_history": crh}
    for s in sels:
        a = acc(debates, s)
        lo0, hi0 = boot_delta(debates, s, "round0_majority")
        lof, hif = boot_delta(debates, s, "final_majority")
        f0 = "*" if lo0 > 0 else ("-" if hi0 < 0 else " ")
        ff = "*" if lof > 0 else ("-" if hif < 0 else " ")
        print(f"  {s:<24}{a:.3f}   vs round0 {a - r0:+.3f} [{lo0:+.3f},{hi0:+.3f}]{f0}"
              f"   vs finalmaj {a - fm:+.3f} [{lof:+.3f},{hif:+.3f}]{ff}")
        res[s] = {"acc": a, "d_round0": a - r0, "ci_round0": [lo0, hi0], "beats_round0": lo0 > 0,
                  "d_finalmaj": a - fm, "ci_finalmaj": [lof, hif], "beats_finalmaj": lof > 0}
    return res


def main():
    debates = build_debates()
    learned = add_learned(debates)
    out, beat0 = {}, defaultdict(int)
    for bench in BENCHES:
        bd = [d for d in debates if d["bench"] == bench]
        if bd:
            out[bench] = summarize(bd, bench, learned)
            for s in learned:
                if out[bench][s]["d_round0"] > 0:
                    beat0[s] += 1
    out["pooled"] = summarize(debates, "POOLED", learned)
    print("\n  --- vs round0 (the baseline to beat): pooled CI>0 AND >=3/5 benches ---")
    for s in learned:
        p = out["pooled"][s]
        go = p["beats_round0"] and beat0[s] >= 3
        sqa = out["strategyqa"][s]["d_finalmaj"] if "strategyqa" in out else float("nan")
        print(f"  {s:<24} pooled vs round0 {p['d_round0']:+.3f} {'sig' if p['beats_round0'] else 'ns'}"
              f" | benches>round0 {beat0[s]}/5 | strategyqa vs finalmaj {sqa:+.3f} -> {'GO' if go else 'no'}")
        out["pooled"][s]["GO_vs_round0"] = bool(go)
        out["pooled"][s]["benches_beat_round0"] = beat0[s]
    out["_note"] = ("Convergence-gated provenance: credit late consensus only when reasoning genuinely "
                    "converges (jaccard residualized vs debate baseline, per E3b surface-adoption). "
                    "Primary = beat round0_majority pooled (CI>0, >=3/5 benches) AND fix StrategyQA.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
