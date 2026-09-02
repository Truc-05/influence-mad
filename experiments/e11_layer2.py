"""
E11 Layer 2 — per-benchmark controller VERDICT (ZERO-GPU given a per-seed cache).
The decisive test after the per-seed fix: does hazard DETECTABILITY convert into
intervention VALUE on a new benchmark?

For each benchmark we report the four headline quantities the project requires, each
with a cluster-bootstrap over questions (the resampling unit; seeds of a question are
correlated repeats):
    (1) Δacc hazard − random   @ 20/30/40% budget and AUC[20-40]   <- detector value (KEY)
    (2) Δacc hazard − always   @ 40% budget                        <- does targeting lose?
    (3) intervention rate at which hazard first reaches always-on   <- efficiency
    (4) rescue (help@would-die) vs harm (harm@would-live)          <- the asymmetry

GO / NO-GO (StrategyQA): GO to a full 6-benchmark expansion iff the AUC[20-40]
hazard−random 95% CI excludes 0 (the detector significantly beats a random budget).
If it fails even on StrategyQA (the strongest Layer-1 detectability, ROC 0.87), the
bottleneck is the controller policy, not model generalization -> fix the policy before
spending GPU on 6 models.

Usage:
  python experiments/e11_layer2.py                 # gsm8k+bbh sanity (should match E9)
  python experiments/e11_layer2.py --bench strategyqa \
         --cache results/cache/e6_sqa_perseed.pkl   # once its precompute lands
Output: results/tables/e11_layer2_<bench>.json (+ appended cross-benchmark figure)
"""
import os, sys, json, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from collections import defaultdict
import experiments.e6_offline as off

BUDGETS = [0.20, 0.30, 0.40]
B = 4000
ALL = ("f1_", "f2_", "f3_", "f4_")


def topk_fire(score, k, rng):
    n = len(score)
    order = np.lexsort((rng.random(n), -score))
    fire = np.zeros(n, bool); fire[order[:k]] = True
    return fire


def metrics_on(idx, h, yc, yk, rng):
    h, yc, yk = h[idx], yc[idx], yk[idx]
    n = len(idx)
    never, always = float(yc.mean()), float(yk.mean())
    out = {"never": never, "always": always}
    for b in BUDGETS:
        k = int(round(b * n))
        hz = float(np.mean(np.where(topk_fire(h, k, rng), yk, yc)))
        rd = float(np.mean(np.where(topk_fire(np.zeros(n), k, rng), yk, yc)))
        out[f"hz_rand@{b}"] = hz - rd
        if b == 0.40:
            out["hz_always@40"] = hz - always
    grid = np.round(np.arange(0.20, 0.401, 0.04), 3)
    hz = [float(np.mean(np.where(topk_fire(h, int(round(b * n)), rng), yk, yc))) for b in grid]
    rd = [float(np.mean(np.where(topk_fire(np.zeros(n), int(round(b * n)), rng), yk, yc))) for b in grid]
    out["auc_hz_rand"] = float(np.mean(hz)) - float(np.mean(rd))
    wl = yc.astype(bool)
    out["harm"] = float(np.mean(1 - yk[wl])) if wl.any() else 0.0
    out["rescue"] = float(np.mean(yk[~wl])) if (~wl).any() else 0.0
    return out


def verdict(events, label, out_path=None):
    h, _ = off.oof_hazard(events, ALL)
    yc, yk = off.outcome_vectors(events)
    clusters = defaultdict(list)
    for i, e in enumerate(events):
        clusters[(e["bench"], e["qid"])].append(i)
    ckeys = list(clusters.values())
    rng = np.random.default_rng(0)

    # stable point estimate (avg over tie-break draws)
    full = np.arange(len(events))
    acc = None
    for _ in range(200):
        s = metrics_on(full, h, yc, yk, rng)
        if acc is None:
            acc = {k: [] for k in s}
        for k in acc:
            acc[k].append(s[k])
    point = {k: float(np.mean(v)) for k, v in acc.items()}
    # cluster bootstrap
    draws = {k: np.empty(B) for k in point}
    for b in range(B):
        pick = rng.integers(0, len(ckeys), len(ckeys))
        idx = np.concatenate([ckeys[j] for j in pick])
        s = metrics_on(idx, h, yc, yk, rng)
        for k in point:
            draws[k][b] = s[k]
    ci = {k: {"point": point[k], "lo": float(np.percentile(draws[k], 2.5)),
              "hi": float(np.percentile(draws[k], 97.5))} for k in point}

    # efficiency: smallest budget at which hazard reaches always-on accuracy
    always = point["always"]
    eff_rate = 1.0
    for b in np.round(np.arange(0.05, 1.001, 0.05), 3):
        a = np.mean([float(np.mean(np.where(topk_fire(h, int(round(b * len(h))), rng), yk, yc)))
                     for _ in range(20)])
        if a >= always - 1e-9:
            eff_rate = float(b); break

    res = {"label": label, "n": len(events), "n_questions": len(ckeys),
           "prevalence_ext": float(np.mean([e["is_ext"] for e in events])),
           "reach_always_at_rate": eff_rate, "ci": ci}
    if out_path:
        json.dump(res, open(out_path, "w"), indent=2, default=float)

    def sig(d):
        return "  *" if (d["lo"] > 0 or d["hi"] < 0) else "   "
    print(f"\n=== {label}  (n={res['n']}, {res['n_questions']} questions, "
          f"prev={res['prevalence_ext']:.2f}, never={ci['never']['point']:.3f}, "
          f"always={ci['always']['point']:.3f}) ===")
    for b in BUDGETS:
        d = ci[f"hz_rand@{b}"]
        print(f"  Δ hazard−random @{int(b*100)}%   {d['point']:+.3f} "
              f"[{d['lo']:+.3f},{d['hi']:+.3f}]{sig(d)}")
    d = ci["auc_hz_rand"]
    print(f"  AUC[20-40] hazard−random  {d['point']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]{sig(d)}"
          f"   <- GO/NO-GO gate")
    d = ci["hz_always@40"]
    print(f"  Δ hazard−always @40%      {d['point']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]{sig(d)}")
    print(f"  reaches always-on acc at {eff_rate:.0%} intervention budget")
    r, hm = ci["rescue"], ci["harm"]
    asym = "  * (non-overlap)" if r["lo"] > hm["hi"] else "   (overlap)"
    print(f"  rescue {r['point']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]  vs  "
          f"harm {hm['point']:.3f} [{hm['lo']:.3f},{hm['hi']:.3f}]{asym}")
    go = ci["auc_hz_rand"]["lo"] > 0
    print(f"  --> detector beats random (AUC CI>0): {'GO' if go else 'not yet'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--sample-live", type=int, default=200)
    args = ap.parse_args()
    os.makedirs("results/tables", exist_ok=True)
    if args.bench:
        evs = off.load_events(benches=[args.bench], sample_live=args.sample_live,
                              cache_path=args.cache or off.CACHE)
        verdict(evs, args.bench, f"results/tables/e11_layer2_{args.bench}.json")
    else:
        evs = off.load_events()          # gsm8k+bbh sanity (should match E9)
        for b in ("gsm8k", "bbh"):
            verdict([e for e in evs if e["bench"] == b], b,
                    f"results/tables/e11_layer2_{b}.json")


if __name__ == "__main__":
    main()
