"""
E16 — dose-response probe EVALUATION (ZERO-GPU). See paper/E16_dose_response_prereg.md.

Loads the frozen E13 population + per-seed ITE labels and the E16 probe responses, builds
the 6 placebo-subtracted Delta-z features (LOCKED, gold-free), learns the T-learner CATE
exactly as E13 (only the feature matrix changes), and applies the frozen gate:
    PASS  <=>  pooled auc_cate_rand.lo > 0   (band [0.20,0.60], cluster-bootstrap over qid).
Also reports the deployable P1-only variant, secondary rho/AUC, per-bench, and the cost frontier.

Usage:
  python experiments/e16_eval.py --selftest    # fabricate signal/null caches, check gate logic (NO GPU)
  python experiments/e16_eval.py               # real eval (needs results/cache/e16_probe.pkl)
"""
import os, sys, json, argparse, pickle
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import experiments.e5_counterfactual as e5           # _norm
import experiments.e6_offline as off
import experiments.e13_benefit_aware as e13
import experiments.e16_probe as e16p                 # _key, _valid, ARMS

OUT = "results/tables/e16_dose_response.json"
CACHE = "results/cache/e16_probe.pkl"
DZ_KEYS = ["d_flip", "d_diff_flip", "d_maxclust", "d_peeralign", "d_rea_lex", "d_rea_len"]
P1_KEYS = ["p1_flip", "p1_maxclust", "p1_peeralign", "p1_rea_lex", "p1_rea_len"]


def _toks(s):
    return set((s or "").lower().split())


def _jacc_dist(a, b):
    A, B = _toks(a), _toks(b)
    if not A and not B:
        return 0.0
    return 1.0 - len(A & B) / len(A | B)


def _arm_stats(arm, trig, names):
    """Per-arm scalars + per-agent flip flags, all gold-free."""
    a0 = {n: e5._norm(trig[n]["answer"]) for n in names}
    m = max(a0.values(), key=list(a0.values()).count)      # peer-majority @ trigger
    flips = {n: (e5._norm(arm[n]["answer"]) != a0[n]) for n in names}
    ans = [e5._norm(arm[n]["answer"]) for n in names]
    maxclust = max(ans.count(x) for x in set(ans)) / len(names)
    peeralign = np.mean([e5._norm(arm[n]["answer"]) == m for n in names])
    rea_lex = np.mean([_jacc_dist(arm[n]["reasoning"], trig[n]["reasoning"]) for n in names])
    rea_len = np.mean([abs(len(_toks(arm[n]["reasoning"])) - len(_toks(trig[n]["reasoning"])))
                       for n in names])
    return {"flip": float(np.mean(list(flips.values()))), "maxclust": float(maxclust),
            "peeralign": float(peeralign), "rea_lex": float(rea_lex),
            "rea_len": float(rea_len)}, flips


def features_for(ev, resp):
    """Return (dz_vec[6], p1_vec[5]) for one event from its P0/P1 responses."""
    names, trig = ev["names"], ev["trigger_state"]
    s0, f0 = _arm_stats(resp["P0"], trig, names)
    s1, f1 = _arm_stats(resp["P1"], trig, names)
    diff_flip = (np.mean([f1[n] and not f0[n] for n in names])
                 - np.mean([f0[n] and not f1[n] for n in names]))
    dz = [s1["flip"] - s0["flip"], float(diff_flip), s1["maxclust"] - s0["maxclust"],
          s1["peeralign"] - s0["peeralign"], s1["rea_lex"] - s0["rea_lex"],
          s1["rea_len"] - s0["rea_len"]]
    p1 = [s1["flip"], s1["maxclust"], s1["peeralign"], s1["rea_lex"], s1["rea_len"]]
    return np.array(dz, float), np.array(p1, float)


def build_matrix(events, cache):
    keep, X_dz, X_p1 = [], [], []
    miss = 0
    for ev in events:
        v = cache.get(e16p._key(ev))
        if not e16p._valid(v):
            miss += 1
            continue
        dz, p1 = features_for(ev, v)
        keep.append(ev); X_dz.append(dz); X_p1.append(p1)
    return keep, np.array(X_dz), np.array(X_p1), miss


def cate_T(X, events):
    """T-learner tau = p1(x)-p0(x), cross-fit by qid (identical to E13)."""
    groups = np.array([e["qid"] for e in events])
    y0 = np.array([int(bool(e["control"]["rec_maj"])) for e in events])
    y1 = np.array([int(bool(e["combined"]["rec_maj"])) for e in events])
    p0 = e13._oof_clf(X, y0, groups)
    p1 = e13._oof_clf(X, y1, groups)
    return p1 - p0, y0, y1


def eval_scope(events, X_dz, X_p1, label):
    yc, yk = off.outcome_vectors(events)
    ite = (yk.astype(int) - yc.astype(int)).astype(float)
    never = float(yc.mean())
    tau, y0, y1 = cate_T(X_dz, events)
    tau_p1, _, _ = cate_T(X_p1, events)
    haz, _ = off.oof_hazard(events)
    S = {"random": np.zeros(len(events)),
         "disagreement": np.array([e["feat"]["f1_disagree_ratio"] for e in events], float),
         "hazard (which fails)": haz,
         "CATE (which helps)": tau,
         "oracle (true ITE)": ite}
    deltas = e13.bootstrap_deltas(events, S, yc, yk)               # band [0.20,0.60], qid bootstrap, B=4000
    # deployable P1-only variant through the same estimator
    S_p1 = dict(S); S_p1["CATE (which helps)"] = tau_p1
    deltas_p1 = e13.bootstrap_deltas(events, S_p1, yc, yk)
    diag = {
        "auc_effect": float(roc_auc_score((ite != 0).astype(int), np.abs(tau)))
                      if 0 < (ite != 0).mean() < 1 else float("nan"),
        "auc_ite_sign": float(roc_auc_score((ite[ite != 0] > 0).astype(int), tau[ite != 0]))
                        if (ite != 0).sum() > 1 and len(set((ite[ite != 0] > 0))) == 2 else float("nan"),
        "rho_T_ite": float(spearmanr(tau, ite).correlation),
        "rho_p1_ite": float(spearmanr(tau_p1, ite).correlation),
    }
    rng = np.random.default_rng(0)
    cost = {"control_calls": float(np.mean([e["control"]["calls"] for e in events])),
            "combined_calls": float(np.mean([e["combined"]["calls"] for e in events])),
            "probe_calls_per_event": 6.0}    # P0+P1 = 2 arms * 3 agents
    return {"label": label, "n": len(events),
            "n_questions": len({e["qid"] for e in events}), "never": never,
            "diagnostics": diag,
            "delta_dz": deltas["auc_cate_rand"], "delta_p1": deltas_p1["auc_cate_rand"],
            "budget_deltas": {k: deltas[k] for k in deltas if k.startswith("cate_rand@")},
            "oracle_gain@40": float(off.acc_at_budget(ite, yc, yk, 0.40, rng)[0] - never),
            "cost": cost}


def report(res):
    for s in ("pooled", "gsm8k", "bbh"):
        if s not in res:
            continue
        r = res[s]; g = r["diagnostics"]
        dz, p1 = r["delta_dz"], r["delta_p1"]
        sig = "  *PASS*" if dz["lo"] > 0 else ""
        print(f"\n=== {s}  (n={r['n']}, {r['n_questions']} q, never={r['never']:.3f}) ===")
        print(f"  Delta-z  AUC[20-60] CATE-random {dz['point']:+.3f} [{dz['lo']:+.3f},{dz['hi']:+.3f}]{sig}")
        print(f"  P1-only  AUC[20-60] CATE-random {p1['point']:+.3f} [{p1['lo']:+.3f},{p1['hi']:+.3f}]"
              f"{'  (deployable-sig)' if p1['lo']>0 else ''}")
        print(f"  rho(tau,ITE) dz={g['rho_T_ite']:+.2f} p1={g['rho_p1_ite']:+.2f} | "
              f"AUC(effect)={g['auc_effect']:.2f} AUC(ITE-sign)={g['auc_ite_sign']:.2f} "
              f"(sign>0 predicted)")
        c = r["cost"]
        print(f"  cost: probe {c['probe_calls_per_event']:.0f} + always(combined) "
              f"{c['combined_calls']:.1f} vs never(control) {c['control_calls']:.1f} calls/event")
    pooled = res.get("pooled", {})
    verdict = "PASS (H_response supported)" if pooled.get("delta_dz", {}).get("lo", -1) > 0 \
        else "FAIL -> STOP per prereg §9 (one future-work sentence; keep triple-negative)"
    print(f"\n  PRIMARY GATE (pooled Delta-z AUC[20-60] CI-lo > 0): {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    events_all = off.load_events()
    cache = pickle.load(open(args.cache, "rb"))
    events, X_dz, X_p1, miss = build_matrix(events_all, cache)
    print(f"loaded {len(events)}/{len(events_all)} events with valid probe responses "
          f"({miss} missing -> generate them first)")
    assert miss == 0, "incomplete probe cache; run experiments/e16_probe.py to completion"
    idx = {s: [i for i, e in enumerate(events) if e["bench"] == s] for s in ("gsm8k", "bbh")}
    res = {"pooled": eval_scope(events, X_dz, X_p1, "pooled")}
    for s, ii in idx.items():
        if ii:
            res[s] = eval_scope([events[i] for i in ii], X_dz[ii], X_p1[ii], s)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    report(res)
    print(f"\n[OK] {OUT}")


# ------------------------------- selftest (NO GPU) ----------------------------
def _fake_cache(events, mode, seed=0):
    """Fabricate P0/P1 responses. mode='signal': P1 moves iff ITE>0 (placebo static) ->
    d_diff_flip must correlate + with ITE. mode='null': random moves uncorrelated with ITE."""
    rng = np.random.default_rng(seed)
    cache = {}
    for e in events:
        ite = int(bool(e["combined"]["rec_maj"])) - int(bool(e["control"]["rec_maj"]))
        cache[e16p._key(e)] = {"P0": {}, "P1": {}}
        for n in e["names"]:
            base = e["trigger_state"][n]
            if mode == "signal":
                p1_move = ite > 0 or (ite == 0 and rng.random() < 0.2)
                p0_move = rng.random() < 0.2
            else:
                p1_move = rng.random() < 0.4
                p0_move = rng.random() < 0.4
            cache[e16p._key(e)]["P0"][n] = {
                "answer": "X" if p0_move else base["answer"],
                "reasoning": "alt reasoning" if p0_move else base["reasoning"]}
            cache[e16p._key(e)]["P1"][n] = {
                "answer": "X" if p1_move else base["answer"],
                "reasoning": "alt reasoning" if p1_move else base["reasoning"]}
    return cache


def selftest():
    events_all = off.load_events()
    for mode in ("signal", "null"):
        cache = _fake_cache(events_all, mode)
        events, X_dz, X_p1, miss = build_matrix(events_all, cache)
        assert miss == 0 and X_dz.shape == (len(events), 6) and X_p1.shape == (len(events), 5)
        r = eval_scope(events, X_dz, X_p1, "pooled")
        dz = r["delta_dz"]; rho = r["diagnostics"]["rho_T_ite"]
        print(f"[selftest:{mode}] n={r['n']} feats dz=6 p1=5 | "
              f"pooled Delta-z {dz['point']:+.3f} [{dz['lo']:+.3f},{dz['hi']:+.3f}]  rho={rho:+.2f}")
        if mode == "signal":
            assert rho > 0.05, f"signal cache should give +rho, got {rho}"
        # gate machinery executed & produced a finite CI in both modes
        assert np.isfinite(dz["lo"]) and np.isfinite(dz["hi"])
    print("[selftest] Delta-z extraction (6) + P1-only (5) + T-learner + frozen gate "
          "run end-to-end; +signal gives +rho. PLUMBING VALID (NO GPU).")


if __name__ == "__main__":
    main()
