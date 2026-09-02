"""
E17-precheck — Trajectory benefit-identifiability (ZERO-GPU go/no-go).

HYPOTHESIS (new, NOT an E13-E16 rescue): treatment benefit may be UNIDENTIFIABLE at t=0
(E13 cheap / E14 semantic / E15 probe / E16 dose-response all failed the ex-ante gate) but
become IDENTIFIABLE as the natural debate EVOLVES. This is a different scientific question:
not "before intervening, do I know who benefits?" but "when in the trajectory does enough
information appear to make intervention actionable?"

METHOD: re-run the FROZEN E13 CATE pipeline UNCHANGED — same ITE target (Y1-Y0 =
combined.rec_maj - control.rec_maj), same cross-fit T-learner (e13._oof_clf, tau_T=p1-p0),
same cluster-boot policy-value gate (e13.bootstrap_deltas, band [0.20,0.60], qid clusters,
B=4000, PASS iff auc_cate_rand.lo>0) — but feed features observed up to round r_obs of the
NATURAL (control-arm) trajectory, r_obs in {0,1,2}. Features come from the SAME round-parametric
machinery e3._features(p, r_obs, infl); ONLY r_obs changes. r_obs=0 must reproduce frozen E13.

LEAKAGE CAVEAT (decisive for reading the result): the control arm IS the natural debate, so
Y0 = its final outcome. By r=2 the trajectory nearly DETERMINES Y0, so effect-predictability
can rise MECHANICALLY (knowing Y0, not benefit-relevant dynamics). Therefore:
  * DECISIVE, leak-robust read = r=1  (round 2 not yet observed; Y0 not yet fixed).
  * r=2 = OPTIMISTIC UPPER BOUND; report auc_predict_Y0 per round as the leak thermometer.
Also causally: natural rounds 1-2 are POST-gate (intervention t_star=1 would have overwritten
them), so a PASS here only motivates a GPU study that generates intervene-at-t counterfactuals.
This precheck answers the cheaper question: does the SIGNAL exist in the trajectory at all?

DECISION: if r=1 lifts sign toward positive (spearman_T_ite>0) and moves auc_cate_rand.lo
toward/above 0 vs r=0 -> GO build the sequential controller. If r=0~r=1 stay at chance and only
r=2 improves alongside auc_predict_Y0 -> ~1 -> mechanical leak -> NO-GO, go generalization+submit.

ZERO-GPU. Run:
  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e17_trajectory_precheck.py
"""
import os, sys, glob, json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from src.utils.io_utils import read_jsonl
import experiments.e3_router as e3
import experiments.e6_offline as off
import experiments.e13_benefit_aware as e13

ROUNDS = [0, 1, 2]
OUT = "results/tables/e17_trajectory_precheck.json"
GRID = np.round(np.arange(0.20, 0.601, 0.05), 3)   # same band bootstrap_deltas averages


def per_round_features(events):
    """(bench:qid:seed) -> {r_obs: feat_dict} for r_obs in ROUNDS, built with the EXACT
    build_population influence + prep_log path (e6_controller.build_population:64-89), so
    feat[r=0] is identical to the frozen event['feat']. Only r_obs varies."""
    needed = sorted({(e["bench"], e["seed"]) for e in events})
    featR, clamped = {}, 0
    for bench, seed in needed:
        f = f"data/debate_logs/{bench}_seed{seed}.jsonl"
        logs = read_jsonl(f)
        agents = list(logs[0]["final_answers"].keys())
        M = e3.build_influence_matrix(logs, agents)
        infl = {"scores": e3.compute_influence_scores(M, agents),
                "stub": e3._global_stub(logs, agents), "M": M,
                "idx": {a: i for i, a in enumerate(agents)}}
        for lg in logs:
            p = e3.prep_log(lg, bench)
            p["_changed"] = [None] + [lg["rounds"][r].get("changed", {})
                                      for r in range(1, p["F"] + 1)]
            key = f"{bench}:{lg['question_id']}:{seed}"
            fr = {}
            for r in ROUNDS:
                rr = min(r, p["F"])            # clamp for short debates (F<2)
                if rr != r:
                    clamped += 1
                fr[r] = e3._features(p, rr, infl)
            featR[key] = fr
    return featR, clamped


def _oracle_auc(scores, yc, yk):
    """Deterministic oracle ceiling in auc_cate_rand units (single draw, no bootstrap)."""
    rng = np.random.default_rng(0)
    rand = scores["random"]
    return float(np.mean([e13._pv(scores["oracle (true ITE)"], yc, yk, g, rng)
                          - e13._pv(rand, yc, yk, g, rng) for g in GRID]))


def analyze(evs, label):
    print(f"\n=== {label}  (n={len(evs)}, {len({e['qid'] for e in evs})} qid) ===")
    rows = {}
    oracle = None
    for r in ROUNDS:
        for e in evs:
            e["feat"] = e[f"feat{r}"]                 # swap observation point -> reuse E13 verbatim
        scores, tau_T, ite, diag = e13.scores_for(evs)
        yc, yk = off.outcome_vectors(evs)
        bd = e13.bootstrap_deltas(evs, scores, yc, yk)
        if oracle is None:
            oracle = _oracle_auc(scores, yc, yk)
        cr = bd["auc_cate_rand"]
        rows[r] = {"auc_effect": diag["auc_predict_effect"],
                   "spearman_T": diag["spearman_T_ite"],
                   "spearman_S": diag["spearman_S_ite"],
                   "auc_Y0_leak": diag["auc_predict_Y0"],
                   "auc_Y1": diag["auc_predict_Y1"],
                   "auc_cate_rand": cr,
                   "cate_rand@0.4": bd["cate_rand@0.4"]}
        pass_gate = "PASS" if cr["lo"] > 0 else "fail"
        print(f"  r={r}: effAUC {diag['auc_predict_effect']:.3f} | "
              f"rho_T {diag['spearman_T_ite']:+.3f} | "
              f"cate-rand {cr['point']:+.4f} [{cr['lo']:+.4f},{cr['hi']:+.4f}] {pass_gate} | "
              f"leak Y0-AUC {diag['auc_predict_Y0']:.3f} Y1-AUC {diag['auc_predict_Y1']:.3f}")
    print(f"  oracle ceiling (auc_cate_rand units, constant): {oracle:+.4f}")
    return {"oracle_auc_cate_rand": oracle, "by_round": rows}


def main():
    events = off.load_events()
    featR, clamped = per_round_features(events)
    for e in events:
        fr = featR[f"{e['bench']}:{e['qid']}:{e['seed']}"]
        for r in ROUNDS:
            e[f"feat{r}"] = fr[r]

    # sanity: recomputed r=0 features must equal the frozen event['feat']
    keys = off.feature_keys(events, e13.ALL)
    mism = sum(1 for e in events for k in keys if abs(e["feat0"][k] - e["feat"][k]) > 1e-9)
    print(f"population {len(events)} events; feat0-vs-frozen mismatches={mism} "
          f"(expect 0); clamped-round features={clamped}")

    out = {"n": len(events), "feat0_mismatches": mism, "clamped": clamped,
           "note": "r=1 is the leak-robust decisive read; r=2 is an optimistic upper bound "
                   "(auc_Y0_leak is the Y0-leakage thermometer). GPU intervene-at-t "
                   "counterfactuals needed only if r=1 shows a real lift."}
    by_bench = {e["bench"] for e in events}
    out["pooled"] = analyze(events, "pooled")
    for b in sorted(by_bench):
        out[b] = analyze([e for e in events if e["bench"] == b], b)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
