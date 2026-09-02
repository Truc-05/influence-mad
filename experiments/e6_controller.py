"""
E6 — hazard-triggered adaptive controller (offline-policy evaluation).

Idea (from paper/E6_design.md): decouple the expensive part (GPU re-query) from the
cheap part (policy sweep). At a single early gate g=0 (observe round 0, gate the
round-1 revision onward — covers all death timings t*∈{1,2}), precompute TWO
counterfactuals per debate:
    control  = plain re-query forward from round 1 (E5 stochastic baseline)
    combined = self_verify + influence_resistant re-query forward from round 1
Then every controller is an OFFLINE selection policy over the two stored outcomes:
    never        -> control everywhere              (accuracy floor, 0% intervention)
    always       -> combined everywhere             (100% intervention)
    hazard(tau)  -> combined iff E3a hazard@g >= tau
    random(p)    -> combined on a random p-fraction  (matched-budget control)
Sweep tau -> accuracy-vs-intervention-rate curve; the detector's value is the gap
between the hazard curve and the random-matched curve; collateral is decomposed on
would-live (control already correct) vs would-die (control wrong) debates.

Population: debates with a correct answer alive at round 0 (something to protect).
E3a hazard = leave-one-group-out OOF Stage-A score from round-0 features.

Reuses e5_counterfactual.run_arm via gate->(t_star=g+1, trigger_state=logged@g), and
e3_router feature/influence machinery. GPU (Ollama) only for the precompute; everything
else is CPU. Run generation against the gpu1-pinned server (OLLAMA on :11435).

Usage:
  python experiments/e6_controller.py --selftest              # stub gen, NO GPU
  python experiments/e6_controller.py --smoke 6 --bench gsm8k # tiny real precompute
  python experiments/e6_controller.py --bench gsm8k,bbh --sample-live 200
Output: results/tables/e6_controller.json  (+ precompute cache)
"""
import argparse, glob, json, os, pickle, sys
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.utils.io_utils import read_jsonl, load_config
from src.agents.base_agent import AgentResponse
import experiments.e3_router as e3
import experiments.e5_counterfactual as e5

GATE = 0                    # observe rounds[0..GATE], gate the revision GATE+1 onward
CACHE = "results/cache/e6_precompute.pkl"
OUT = "results/tables/e6_controller.json"


# ----------------------------- population + hazard ----------------------------
def build_population(benches, sample_live, seed=0):
    """Debates with a correct answer alive at the gate. Each -> an event dict for
    run_arm (trigger_state = logged answers @ gate; t_star = gate+1) + round-0
    hazard features + is_extinction label. Would-live debates are subsampled."""
    rng = np.random.default_rng(seed)
    events = []
    for bench in benches:
        pos, live = [], []
        for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
            seed = int(f.split("_seed")[1].split(".")[0])   # per-seed identity for the cache key
            logs = read_jsonl(f)
            if not logs:
                continue
            agents = list(logs[0]["final_answers"].keys())
            M = e3.build_influence_matrix(logs, agents)
            infl = {"scores": e3.compute_influence_scores(M, agents),
                    "stub": e3._global_stub(logs, agents), "M": M,
                    "idx": {a: i for i, a in enumerate(agents)}}
            for lg in logs:
                p = e3.prep_log(lg, bench)
                p["_changed"] = [None] + [lg["rounds"][r].get("changed", {})
                                          for r in range(1, p["F"] + 1)]
                if p["ncorr"][GATE] < 1:
                    continue                                   # nothing correct to protect
                ext = e3.extinction_of(p)
                trigger_state = {n: {"answer": lg["rounds"][GATE]["answers"][n]["answer"],
                                     "reasoning": lg["rounds"][GATE]["answers"][n]["reasoning"]}
                                 for n in p["agents"]}
                ev = {"qid": lg["question_id"], "bench": bench, "seed": seed,
                      "question": lg["question"],
                      "gt": lg["ground_truth"], "ca": lg.get("correct_answers"),
                      "names": p["agents"], "trigger_state": trigger_state,
                      "t_star": GATE + 1, "is_ext": ext is not None,
                      "feat": e3._features(p, GATE, infl)}
                (pos if ev["is_ext"] else live).append(ev)
        if sample_live and len(live) > sample_live:
            live = [live[i] for i in rng.choice(len(live), sample_live, replace=False)]
        events += pos + live
        print(f"  [{bench}] {len(pos)} extinction + {len(live)} would-live (sampled) = {len(pos)+len(live)}")
    return events


def oof_hazard(events):
    """Leave-one-group-out OOF Stage-A hazard from round-0 features (leak-free)."""
    keys = [k for k in events[0]["feat"] if k.startswith(("f1_", "f2_", "f3_", "f4_"))]
    X = np.array([[e["feat"][k] for k in keys] for e in events])
    y = np.array([1 if e["is_ext"] else 0 for e in events])
    groups = np.array([e["qid"] for e in events])
    ns = min(5, np.unique(groups).size, int(y.sum()), int((1 - y).sum()))
    haz = np.full(len(y), y.mean())
    if ns >= 2:
        for tr, te in GroupKFold(ns).split(X, y, groups):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X[tr], y[tr]); haz[te] = m.predict_proba(X[te])[:, 1]
    for e, h in zip(events, haz):
        e["hazard"] = float(h)
    return float(y.mean())


# ------------------------------ counterfactuals -------------------------------
def precompute(events, config, gen_fn, cache_path):
    """Run control + combined re-query forward for every event (the GPU step).
    Cached by qid so it resumes. Stores rec_maj/rec_any/preserved/calls per arm."""
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    agents = e5.build_agents(config)
    mr = config["debate"]["max_rounds"]
    done = 0
    for i, ev in enumerate(events):
        key = f"{ev['bench']}:{ev['qid']}:{ev['seed']}"   # per-SEED key (fixes the seed-collapse)
        if key not in cache:
            out = {}
            for arm in ("control", "combined"):
                pres, rmaj, rany, calls = e5.run_arm(ev, arm, agents, mr, gen_fn)
                out[arm] = {"preserved": pres, "rec_maj": rmaj, "rec_any": rany, "calls": calls}
            cache[key] = out
            if (i + 1) % 10 == 0:
                pickle.dump(cache, open(cache_path, "wb"))
        ev.update(cache[key]); done += 1
        if (i + 1) % 20 == 0 or i + 1 == len(events):
            print(f"  precompute [{i+1}/{len(events)}]")
    pickle.dump(cache, open(cache_path, "wb"))
    return events


# --------------------------------- policies -----------------------------------
def _acc_cost(events, decide):
    """decide(ev)->bool (True=apply combined). Returns (accuracy, intervention_rate)."""
    n = len(events)
    correct = sum((ev["combined"] if decide(ev) else ev["control"])["rec_maj"] for ev in events)
    fired = sum(1 for ev in events if decide(ev))
    return correct / n, fired / n


def evaluate(events, n_boot=1000):
    haz = np.array([ev["hazard"] for ev in events])
    taus = np.unique(np.concatenate([[-1.0], np.quantile(haz, np.linspace(0, 1, 21)), [2.0]]))
    rng = np.random.default_rng(0)
    curve = []
    for tau in taus:
        acc, rate = _acc_cost(events, lambda ev: ev["hazard"] >= tau)
        curve.append({"tau": float(tau), "acc": acc, "rate": rate})
    never_acc, _ = _acc_cost(events, lambda ev: False)
    always_acc, _ = _acc_cost(events, lambda ev: True)
    # random-matched: for each achieved rate, random subset of same size, averaged
    rand_curve = []
    order = np.arange(len(events))
    for pt in curve:
        k = round(pt["rate"] * len(events))
        accs = []
        for _ in range(200):
            pick = set(rng.choice(order, k, replace=False).tolist()) if k else set()
            accs.append(sum((events[i]["combined"] if i in pick else events[i]["control"])["rec_maj"]
                            for i in range(len(events))) / len(events))
        rand_curve.append({"rate": pt["rate"], "acc": float(np.mean(accs)),
                           "acc_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]})
    # GainRetention(tau) = (Acc_hazard - Acc_never) / (Acc_always - Acc_never)
    denom = always_acc - never_acc
    gr = lambda a: (a - never_acc) / denom if abs(denom) > 1e-9 else float("nan")
    for h, r in zip(curve, rand_curve):
        h["gain_retention"] = gr(h["acc"]); r["gain_retention"] = gr(r["acc"])
    # operating band 20-40% intervention rate
    op = [{"rate": h["rate"], "hazard_acc": h["acc"], "hazard_GR": gr(h["acc"]),
           "random_acc": r["acc"], "random_GR": gr(r["acc"]),
           "beats_random": h["acc"] > r["acc_ci"][1]}                # above random 95% CI
          for h, r in zip(curve, rand_curve) if 0.20 <= h["rate"] <= 0.40]
    passing = [o for o in op if o["hazard_GR"] >= 0.70 and o["beats_random"]]
    # detector value: at the tau achieving ~30% intervention, hazard_acc - random_acc
    target = min(curve, key=lambda c: abs(c["rate"] - 0.30))
    trand = min(rand_curve, key=lambda c: abs(c["rate"] - target["rate"]))
    # collateral decomposition on the natural population
    wl = [e for e in events if e["control"]["rec_maj"]]      # would-live (control correct)
    wd = [e for e in events if not e["control"]["rec_maj"]]  # would-die
    harm = np.mean([1 - e["combined"]["rec_maj"] for e in wl]) if wl else 0.0
    help_ = np.mean([e["combined"]["rec_maj"] for e in wd]) if wd else 0.0
    return {
        "n": len(events), "prevalence_ext": float(np.mean([e["is_ext"] for e in events])),
        "never_acc": never_acc, "always_acc": always_acc, "gain_denominator": denom,
        "hazard_curve": curve, "random_curve": rand_curve,
        "operating_20_40": op, "PASS": bool(passing),
        "at_30pct": {"rate": target["rate"], "hazard_acc": target["acc"],
                     "random_acc": trand["acc"], "detector_gain": target["acc"] - trand["acc"],
                     "hazard_GR": gr(target["acc"]), "random_GR": gr(trand["acc"])},
        "collateral": {"n_would_live": len(wl), "n_would_die": len(wd),
                       "harm_rate_on_would_live": float(harm), "help_rate_on_would_die": float(help_)},
        "mean_calls": {"control": float(np.mean([e["control"]["calls"] for e in events])),
                       "combined": float(np.mean([e["combined"]["calls"] for e in events]))},
    }


# --------------------------------- selftest -----------------------------------
def selftest():
    """Stub generator (NO GPU): correct-holder debates recover under combined more
    than control; validates policy/curve/collateral plumbing + offline selection."""
    events = build_population(["gsm8k"], sample_live=15, seed=0)
    assert events, "no population"
    base = oof_hazard(events)
    # deterministic stub outcomes: combined recovers ext events, control mostly doesn't
    rng = np.random.default_rng(1)
    for e in events:
        c = rng.random() < (0.2 if e["is_ext"] else 0.7)     # control rec_maj
        k = c or (e["is_ext"] and rng.random() < 0.6)        # combined recovers more of ext
        e["control"] = {"rec_maj": bool(c), "rec_any": bool(c), "preserved": bool(c), "calls": 6}
        e["combined"] = {"rec_maj": bool(k), "rec_any": bool(k), "preserved": bool(k), "calls": 6}
    res = evaluate(events)
    assert 0.0 <= res["at_30pct"]["rate"] <= 1.0
    assert res["never_acc"] <= res["always_acc"] + 1e-9
    print(f"[selftest] n={res['n']} ext_prev={res['prevalence_ext']:.2f} base_haz={base:.2f}")
    print(f"[selftest] never={res['never_acc']:.2f} always={res['always_acc']:.2f} "
          f"@30%: hazard={res['at_30pct']['hazard_acc']:.2f} random={res['at_30pct']['random_acc']:.2f} "
          f"gain={res['at_30pct']['detector_gain']:+.2f}")
    print(f"[selftest] collateral: harm@live={res['collateral']['harm_rate_on_would_live']:.2f} "
          f"help@die={res['collateral']['help_rate_on_would_die']:.2f}")
    print("[selftest] policy/curve/collateral plumbing VALID (no GPU).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="gsm8k,bbh")
    ap.add_argument("--sample-live", type=int, default=200)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    benches = args.bench.split(",")
    events = build_population(benches, sample_live=args.sample_live)
    if args.smoke:
        events = events[:args.smoke]
    base = oof_hazard(events)
    print(f"population n={len(events)}  ext-prevalence={base:.3f}")
    config = load_config(args.config)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    gen_fn = lambda agent, prompt: agent.generate(prompt)
    events = precompute(events, config, gen_fn, args.cache)
    res = evaluate(events)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    a = res["at_30pct"]
    print(f"\n#### E6 — 4 headline numbers ####")
    print(f"  never (all-control)      acc = {res['never_acc']:.3f}")
    print(f"  always-combined          acc = {res['always_acc']:.3f}")
    print(f"  hazard @~{a['rate']:.0%} intervention  acc = {a['hazard_acc']:.3f}")
    print(f"  random @~{a['rate']:.0%} matched-budget acc = {a['random_acc']:.3f}")
    print(f"  -> GainRetention hazard={a['hazard_GR']:+.2f} vs random={a['random_GR']:+.2f} "
          f"(detector gain {a['detector_gain']:+.3f})")
    print(f"\n  operating band 20-40% intervention:")
    for o in res["operating_20_40"]:
        print(f"    rate={o['rate']:.0%}  hazard GR={o['hazard_GR']:+.2f} (acc {o['hazard_acc']:.3f})  "
              f"random GR={o['random_GR']:+.2f}  beats_random={o['beats_random']}")
    print(f"  PRIMARY GATE (GR>=0.70 & beats random in 20-40%): {'PASS' if res['PASS'] else 'FAIL'}")
    c = res["collateral"]
    print(f"\n  collateral: help@would-die={c['help_rate_on_would_die']:.3f} (n={c['n_would_die']})  "
          f"harm@would-live={c['harm_rate_on_would_live']:.3f} (n={c['n_would_live']})")
    print(f"\n[OK] {args.out}")


if __name__ == "__main__":
    main()
