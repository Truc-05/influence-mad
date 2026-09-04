"""
E23 — Frozen PROVENANCE selector for HP-MAD (ZERO-GPU). Preserve -> Track Provenance -> Select.

E22 showed provenance distinguishes correct vs incorrect hypotheses (n_independent AUC 0.82,
n_persist 0.81, n_support 0.86, first_round 0.70; untuned combo 0.83). E23 turns that into a real
"pick ONE per debate" selector and asks the deployable question: does it beat the deployed
final-majority, with NO extra LLM calls / no external judge / no capability mismatch?

Score (minimal, per user): S(h) = w1·n_independent + w2·n_persist + w3·n_support − w4·first_round.
(resurrected / n_late_adopt dropped — near-null in E22.)

Selectors compared (all gold-free, same e3.prep_log correctness):
  final_majority         PRIMARY BASELINE (deployed)
  round0_majority        E19 heuristic (pre-debate independent consensus)
  antirecency            E19 heuristic (trust early)
  history_frequency      E19 heuristic (most votes)
  influence_vote         CONFERENCE baseline (final vote weighted by PageRank influence)
  prov_equalweight       z-scored equal-weight frozen S(h)  (no fitting; normalization is label-free)
  prov_logistic_xfit     LogisticRegression weights, CROSS-FIT by qid (OOF; never fit on test)
  prov_eq_-independent / -persist / -support   ablations of the equal-weight score

PRIMARY: ΔAcc = Acc_selector − Acc_final_majority, cluster-bootstrap by qid, per-bench + pooled.
SECONDARY: % of the oracle ceiling captured = (Acc − final_maj)/(CR_history − final_maj).
GO iff a provenance selector beats final_majority pooled (CI excludes 0) AND ≥3/5 benches same sign.
ZERO-GPU. Run:
  CUDA_VISIBLE_DEVICES="" python experiments/e23_provenance_selector.py
"""
import os, sys, glob, json
from collections import Counter, defaultdict

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
OUT = "results/tables/e23_provenance_selector.json"
B_BOOT = 4000
FEATS = ["n_independent", "n_persist", "n_support", "first_round"]
SIGN = {"n_independent": +1, "n_persist": +1, "n_support": +1, "first_round": -1}
HEUR = ["final_majority", "round0_majority", "antirecency", "history_frequency", "influence_vote"]


def _pick(scores, freq):
    if not scores:
        return None
    return max(scores, key=lambda b: (scores[b], freq[b], str(b)))


def build_debates(benches=None, logdir="data/debate_logs"):
    """Per debate: candidate features + baseline picks + correctness. Influence built per file.
    benches/logdir overridable so E25 reuses this exact selector pipeline on alt-ensemble logs."""
    debates = []
    for bench in (benches or BENCHES):
        for f in sorted(glob.glob(f"{logdir}/{bench}_seed*.jsonl")):
            logs = read_jsonl(f)
            if not logs:
                continue
            agents = list(logs[0]["final_answers"].keys())
            infl = e3.compute_influence_scores(e3.build_influence_matrix(logs, agents), agents)
            for lg in logs:
                p = e3.prep_log(lg, bench)
                F = p["F"]
                cells = [(r, a) for r in range(F + 1) for a in agents if p["B"][r][a] is not None]
                if not cells:
                    continue
                correct = {b for r in range(F + 1) for a in agents
                           if p["B"][r][a] is not None and p["C"][r][a] for b in [p["B"][r][a]]}
                pool = {p["B"][r][a] for r, a in cells}
                freq = Counter(p["B"][r][a] for r, a in cells)
                feats = {}
                for h in pool:
                    pr = sorted({r for r, a in cells if p["B"][r][a] == h})
                    feats[h] = {
                        "n_independent": sum(1 for a in agents if p["B"][0][a] == h),
                        "n_support": freq[h],
                        "n_persist": len(pr),
                        "first_round": pr[0],
                        "antirecency": sum(F + 1 - r for r, a in cells if p["B"][r][a] == h),
                    }
                final_b = Counter(p["B"][F][a] for a in agents if p["B"][F][a] is not None)
                round0_b = Counter(p["B"][0][a] for a in agents if p["B"][0][a] is not None)
                inf_score = defaultdict(float)
                for a in agents:
                    if p["B"][F][a] is not None:
                        inf_score[p["B"][F][a]] += infl.get(a, 0.0)
                picks = {
                    "final_majority": _pick(dict(final_b), freq),
                    "round0_majority": _pick(dict(round0_b), freq),
                    "history_frequency": _pick({h: feats[h]["n_support"] for h in pool}, freq),
                    "antirecency": _pick({h: feats[h]["antirecency"] for h in pool}, freq),
                    "influence_vote": _pick(dict(inf_score), freq) if inf_score else None,
                }
                debates.append({"bench": bench, "qid": lg["question_id"], "pool": list(pool),
                                "feats": feats, "freq": freq, "correct": correct,
                                "CR_history": bool(pool & correct), "picks": picks})
    return debates


def add_provenance_picks(debates):
    """Equal-weight frozen + ablations + cross-fit logistic. Weights never see test labels."""
    # flatten candidates
    flat = [(di, h) for di, d in enumerate(debates) for h in d["pool"]]
    X = np.array([[debates[di]["feats"][h][f] for f in FEATS] for di, h in flat], float)
    y = np.array([int(h in debates[di]["correct"]) for di, h in flat])
    groups = np.array([debates[di]["qid"] for di, h in flat])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    signs = np.array([SIGN[f] for f in FEATS], float)

    def argmax_per_debate(score_by_flat, name):
        by = defaultdict(dict)
        for k, (di, h) in enumerate(flat):
            by[di][h] = score_by_flat[k]
        for di, d in enumerate(debates):
            pick = _pick(by[di], d["freq"])
            d["picks"][name] = pick

    # equal-weight frozen (label-free normalization) + ablations
    argmax_per_debate(Z @ signs, "prov_equalweight")
    for drop in ["n_independent", "n_persist", "n_support"]:
        s = signs.copy(); s[FEATS.index(drop)] = 0.0
        argmax_per_debate(Z @ s, f"prov_eq_-{drop.split('_')[1]}")

    # cross-fit logistic (OOF): fit on train qids, score test qids
    oof = np.full(len(flat), np.nan)
    ns = min(5, len(set(groups)))
    for tr, te in GroupKFold(ns).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    argmax_per_debate(oof, "prov_logistic_xfit")
    return ["prov_equalweight", "prov_eq_-independent", "prov_eq_-persist",
            "prov_eq_-support", "prov_logistic_xfit"]


def acc(debates, sel):
    return float(np.mean([d["picks"].get(sel) in d["correct"] for d in debates]))


def boot_delta(debates, sel, ref="final_majority", B=B_BOOT):
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
    fm = acc(debates, "final_majority")
    crh = float(np.mean([d["CR_history"] for d in debates]))
    head = crh - fm
    print(f"\n=== {label}  (n={len(debates)}, {len({d['qid'] for d in debates})} qid, "
          f"final-maj {fm:.3f}, CR_history {crh:.3f}) ===")
    res = {"n": len(debates), "final_majority": fm, "CR_history": crh}
    for s in sels:
        a = acc(debates, s)
        if s == "final_majority":
            print(f"  {s:<22}{a:.3f}   (baseline)")
            res[s] = {"acc": a}
            continue
        lo, hi = boot_delta(debates, s)
        cap = (a - fm) / head * 100 if head > 0 else float("nan")
        flag = "*" if lo > 0 else ("-" if hi < 0 else " ")
        print(f"  {s:<22}{a:.3f}   Δ {a - fm:+.3f} [{lo:+.3f},{hi:+.3f}]{flag}  cap {cap:.0f}%")
        res[s] = {"acc": a, "delta": a - fm, "ci": [lo, hi], "sig": lo > 0, "cap_pct": cap}
    return res


def main():
    debates = build_debates()
    prov = add_provenance_picks(debates)
    order = HEUR + prov
    out = {}
    per_bench_sig = defaultdict(list)
    for bench in BENCHES:
        bd = [d for d in debates if d["bench"] == bench]
        if bd:
            out[bench] = summarize(bd, bench, order)
            for s in prov:
                if out[bench][s]["delta"] > 0:
                    per_bench_sig[s].append(bench)
    out["pooled"] = summarize(debates, "POOLED", order)
    # GO check
    print("\n  --- GO check (pooled CI>0 AND >=3/5 benches same sign) ---")
    for s in prov:
        p = out["pooled"][s]
        n_pos = len(per_bench_sig[s])
        go = p["sig"] and n_pos >= 3
        print(f"  {s:<22} pooled Δ {p['delta']:+.3f} {'sig' if p['sig'] else 'ns '} | "
              f"benches Δ>0: {n_pos}/5 -> {'GO' if go else 'no'}")
        out["pooled"][s]["GO"] = bool(go)
        out["pooled"][s]["benches_positive"] = n_pos
    out["_note"] = ("Provenance selector = Preserve->Track Provenance->Select, zero extra inference. "
                    "Weights frozen (equal-weight) or cross-fit by qid (never on test). GO iff beats "
                    "deployed final-majority pooled (CI>0) and >=3/5 benches same sign.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
