"""
Offline harness shared by E7/E8/E9/E12 — ZERO-GPU, never touches Ollama.

Rebuilds the exact E6 population (same benches / sample_live / seed), attaches the
frozen precompute cache directly (no build_agents, no gen_fn), and exposes an OOF
Stage-A hazard that can be trained on any subset of feature groups. Everything
downstream is pure offline policy evaluation over the two stored counterfactual
outcomes {control, combined}.

Run `python experiments/e6_offline.py` for a self-check: it (a) reproduces the frozen
E6 never/always/peak numbers, (b) reports how outcomes are shared across seeds of the
same question (cache is keyed bench:qid), and (c) reports which feature groups are
actually live at the gate g=0.
"""
import os, sys, pickle
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import experiments.e6_controller as e6

BENCHES = ["gsm8k", "bbh"]
SAMPLE_LIVE = 200
SEED = 0
CACHE = "results/cache/e6_precompute_perseed.pkl"   # per-SEED cache (seed-collapse fixed)
GROUP_PREFIXES = ("f1_", "f2_", "f3_", "f4_")


def _key(ev):
    return f"{ev['bench']}:{ev['qid']}:{ev['seed']}"


def load_events(benches=BENCHES, sample_live=SAMPLE_LIVE, seed=SEED, cache_path=CACHE):
    """Exact E6 population with cached {control, combined} outcomes attached. No GPU.
    Keyed per (bench,qid,seed): every seed now carries its OWN counterfactual outcome."""
    events = e6.build_population(benches, sample_live=sample_live, seed=seed)
    cache = pickle.load(open(cache_path, "rb"))
    miss = [_key(e) for e in events if _key(e) not in cache]
    assert not miss, f"{len(miss)} events missing from cache, e.g. {miss[:3]}"
    for ev in events:
        ev.update(cache[_key(ev)])
    return events


def feature_keys(events, prefixes=GROUP_PREFIXES):
    return [k for k in events[0]["feat"] if k.startswith(tuple(prefixes))]


def oof_hazard(events, prefixes=GROUP_PREFIXES):
    """Leave-one-group(question)-out OOF Stage-A hazard from the given feature groups.
    Mirrors e6_controller.oof_hazard exactly when prefixes == all four groups."""
    keys = feature_keys(events, prefixes)
    X = np.array([[e["feat"][k] for k in keys] for e in events])
    y = np.array([1 if e["is_ext"] else 0 for e in events])
    groups = np.array([e["qid"] for e in events])
    ns = min(5, np.unique(groups).size, int(y.sum()), int((1 - y).sum()))
    haz = np.full(len(y), y.mean(), dtype=float)
    if ns >= 2:
        for tr, te in GroupKFold(ns).split(X, y, groups):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X[tr], y[tr])
            haz[te] = m.predict_proba(X[te])[:, 1]
    return haz, keys


# --------- policy primitives over the two stored outcomes (matched-budget) ---------
def outcome_vectors(events):
    """Return (y_control, y_combined) boolean arrays of rec_maj for the two arms."""
    yc = np.array([bool(e["control"]["rec_maj"]) for e in events])
    yk = np.array([bool(e["combined"]["rec_maj"]) for e in events])
    return yc, yk


def acc_at_budget(score, yc, yk, budget, higher_is_riskier=True):
    """Intervene (use combined) on the top-`budget` fraction ranked by `score`.
    Deterministic top-k; ties broken by original order. Returns accuracy."""
    n = len(score)
    k = int(round(budget * n))
    order = np.argsort(-score if higher_is_riskier else score, kind="stable")
    fire = np.zeros(n, dtype=bool)
    fire[order[:k]] = True
    return float(np.mean(np.where(fire, yk, yc)))


def gain_retention(acc, never, always):
    d = always - never
    return (acc - never) / d if abs(d) > 1e-12 else float("nan")


def acc_at_budget(score, yc, yk, budget, rng, R=800):
    """Matched-budget accuracy: apply `combined` to the top-`budget` fraction ranked
    by `score` (higher = intervene first). Ties broken at random, averaged over R
    draws -> fair to coarse/discrete heuristics; the constant-score policy reduces to
    the random-matched baseline. Returns (mean, lo95, hi95) over the tie-break draws."""
    score = np.asarray(score, float)
    n = len(score); k = int(round(budget * n))
    if k <= 0:
        a = float(yc.mean()); return a, a, a
    if k >= n:
        a = float(yk.mean()); return a, a, a
    accs = np.empty(R)
    neg = -score
    for r in range(R):
        order = np.lexsort((rng.random(n), neg))     # score desc, ties random
        fire = np.zeros(n, bool); fire[order[:k]] = True
        accs[r] = np.mean(np.where(fire, yk, yc))
    return float(accs.mean()), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def auc_band(score, yc, yk, rng, lo=0.20, hi=0.40, step=0.02, R=400):
    """Mean matched-budget accuracy over the budget band [lo, hi] (trapezoid / grid
    average). A threshold-free summary of a policy in the operating region."""
    grid = np.round(np.arange(lo, hi + 1e-9, step), 4)
    accs = [acc_at_budget(score, yc, yk, b, rng, R=R)[0] for b in grid]
    return float(np.mean(accs))


# --------------------------------- self-check ---------------------------------
def _selfcheck():
    frozen = json.load(open(e6.OUT))
    events = load_events()
    n = len(events)
    yc, yk = outcome_vectors(events)
    never, always = float(yc.mean()), float(yk.mean())

    print(f"population n={n}  (frozen n={frozen['n']})")
    print(f"never  acc = {never:.4f}  (frozen {frozen['never_acc']:.4f})")
    print(f"always acc = {always:.4f}  (frozen {frozen['always_acc']:.4f})")

    haz, keys = oof_hazard(events)          # ALL groups -> should match frozen hazard
    for e, h in zip(events, haz):
        e["hazard"] = float(h)
    res = e6.evaluate(events)
    peak = max(res["hazard_curve"], key=lambda p: p["acc"])
    fpeak = max(frozen["hazard_curve"], key=lambda p: p["acc"])
    print(f"hazard peak acc = {peak['acc']:.4f} @rate {peak['rate']:.4f}  "
          f"(frozen {fpeak['acc']:.4f} @rate {fpeak['rate']:.4f})")
    ok = (n == frozen["n"] and abs(never - frozen["never_acc"]) < 1e-9
          and abs(always - frozen["always_acc"]) < 1e-9
          and abs(peak["acc"] - fpeak["acc"]) < 1e-6)
    print(f"REPRODUCES FROZEN E6: {'YES' if ok else 'NO'}")

    # --- outcome sharing across seeds of the same question (cache keyed bench:qid) ---
    byq = defaultdict(list)
    for i, e in enumerate(events):
        byq[(e["bench"], e["qid"])].append(i)
    seeds_per_q = [len(v) for v in byq.values()]
    print(f"\nunique questions (bench:qid) = {len(byq)};  "
          f"events per question: mean {np.mean(seeds_per_q):.2f}, max {max(seeds_per_q)}")
    # within a question, do all seed-events share identical control/combined outcome?
    shared = sum(1 for v in byq.values()
                 if len({(yc[i], yk[i]) for i in v}) == 1)
    print(f"questions with identical outcome across all their seed-events: "
          f"{shared}/{len(byq)}  (outcome is cached per qid, so this is 100% by construction)")
    haz_var_q = [np.std([haz[i] for i in v]) for v in byq.values() if len(v) > 1]
    print(f"but hazard DOES vary across seeds: mean within-question hazard std = "
          f"{np.mean(haz_var_q):.4f}")

    # --- which feature groups are actually live at the gate g=0 ---------------
    print("\nfeature-group activity at gate g=0 (nonzero fraction, mean|value|):")
    for pref in GROUP_PREFIXES:
        ks = feature_keys(events, (pref,))
        M = np.array([[e["feat"][k] for k in ks] for e in events])
        nz = float(np.mean(np.any(M != 0, axis=0)))   # fraction of cols with any nonzero
        colvar = float(np.mean(M.std(axis=0) > 1e-9)) # fraction of cols with variance
        print(f"  {pref}: {len(ks)} feats, cols-with-variance={colvar:.2f}, "
              f"mean|val|={np.mean(np.abs(M)):.3f}")


if __name__ == "__main__":
    _selfcheck()
