"""
E15 evaluation (ZERO-GPU except a light flaw-text embed) — Probe-CATE vs baselines,
against the LOCKED gate in paper/E15_active_probe_prereg.md. Reuses the existing per-seed
ITE labels; adds features from the diagnostic probe (e15_active_probe.py cache).

Probe-CATE feature set (deployable controller): [X0_cheap  +  probe scalars
(confidence stats, found-flaw count, flaw length)  +  within-fold PCA of residualized
flaw-text embeddings]. Same S/T learners, GroupKFold(5) by qid, no model search.

GATE: PRIMARY (binding) = cluster-bootstrap 95% CI of (Probe − Random) AUC[20-40] > 0.
      SECONDARY (mechanism) = AUC_effect > 0.65 OR rho(tau,ITE) > 0.20.

Usage: python experiments/e15_eval.py   (after e15_active_probe.py completes)
Output: results/tables/e15_probe.json , results/figures/e15_probe.png
"""
import os, sys, json, pickle, re, hashlib
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.e6_offline as off
import experiments.e13_benefit_aware as e13
import experiments.e14_benefit_semantic as e14

PROBE_CACHE = "results/cache/e15_probe.pkl"
OUT = "results/tables/e15_probe.json"
FIG = "results/figures/e15_probe.png"
ALL = ("f1_", "f2_", "f3_", "f4_")
BUDGETS = e13.BUDGETS
CURVE = e13.CURVE
INK = "#2B2B2B"
GREY, PURPLE, TEAL, BLUE, GOLD = "#8C8C8C", "#8172B3", "#20A39E", "#4C72B0", "#C7A008"


def parse_probe(raw):
    if not raw or raw == "__ERROR__":
        return np.nan, 0, ""
    conf = re.search(r"CONFIDENCE[:\s]*([0-9]{1,3})", raw, re.I)
    conf = float(conf.group(1)) if conf else np.nan
    m = re.search(r"FLAW[:\s]*(.+?)(?:\nCONFIDENCE|$)", raw, re.I | re.S)
    flaw = (m.group(1) if m else raw).strip()
    denies = bool(re.match(r"(none|no\s+flaw|no\s+error|n/?a|correct|confident|there\s+is\s+no)",
                           flaw.strip(), re.I)) or len(flaw) < 8
    return conf, (0 if denies else 1), ("" if denies else flaw)


def build_probe_features(events, probe):
    """Per-event probe scalars + residualized mean flaw-embedding (for within-fold PCA)."""
    conf = {}; found = {}; flaw = {}
    for i, e in enumerate(events):
        for a in e["names"]:
            c, f, t = parse_probe(probe.get(f"{e['bench']}:{e['qid']}:{e['seed']}:{a}"))
            conf[(i, a)] = c; found[(i, a)] = f; flaw[(i, a)] = t
    # impute missing confidence with global median of observed
    obs = [v for v in conf.values() if not np.isnan(v)]
    med = float(np.median(obs)) if obs else 50.0
    for k in conf:
        if np.isnan(conf[k]):
            conf[k] = med
    # embed flaw texts (reuse nomic cache, fill on gpu1)
    emap = e14.embed_texts(sorted({t for t in flaw.values()}))
    byq = defaultdict(list)
    for i, e in enumerate(events):
        for a in e["names"]:
            byq[(e["bench"], e["qid"])].append(emap[flaw[(i, a)]])
    centroid = {q: np.mean(v, axis=0) for q, v in byq.items()}

    scal, resid = [], []
    for i, e in enumerate(events):
        cs = [conf[(i, a)] for a in e["names"]]
        fs = [found[(i, a)] for a in e["names"]]
        ls = [len(flaw[(i, a)]) for a in e["names"]]
        c0 = centroid[(e["bench"], e["qid"])]
        fe = [emap[flaw[(i, a)]] - c0 for a in e["names"]]
        scal.append([np.mean(cs), np.min(cs), np.max(cs), np.std(cs),
                     np.sum(fs), np.mean(fs), np.mean(ls), np.max(ls)])
        resid.append(np.mean(fe, axis=0))
    return np.array(scal, float), np.array(resid, float)


def oof_probe_cate(Xdet, resid, y0, y1, ite, groups, seed=0):
    """OOF S/T learner on [deterministic features, within-fold PCA of flaw residuals]."""
    ns = min(5, np.unique(groups).size)
    tauS = np.zeros(len(ite)); p0 = np.full(len(ite), y0.mean(), float)
    p1 = np.full(len(ite), y1.mean(), float)
    if ns >= 2:
        for tr, te in GroupKFold(ns).split(Xdet, ite, groups):
            Ztr, Zte = e14._feat_fold(Xdet, resid, tr, te)   # concat Xdet + within-fold PCA(resid)
            from sklearn.ensemble import HistGradientBoostingRegressor as R, HistGradientBoostingClassifier as C
            rr = R(max_depth=3, max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=seed)
            rr.fit(Ztr, ite[tr]); tauS[te] = rr.predict(Zte)
            for y, pv in ((y1, p1), (y0, p0)):
                if 0 < y[tr].mean() < 1:
                    cc = C(max_depth=3, max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=seed)
                    cc.fit(Ztr, y[tr]); pv[te] = cc.predict_proba(Zte)[:, 1]
    return tauS, (p1 - p0), p0, p1


def run_scope(events, label, probe_scal, probe_resid, sem_tau):
    keys = off.feature_keys(events, ALL)
    Xcheap = np.array([[e["feat"][k] for k in keys] for e in events], float)
    Xdet = np.hstack([Xcheap, probe_scal])
    y0 = np.array([int(bool(e["control"]["rec_maj"])) for e in events])
    y1 = np.array([int(bool(e["combined"]["rec_maj"])) for e in events])
    ite = (y1 - y0).astype(float)
    groups = np.array([e["qid"] for e in events])
    tauS, tauT, p0, p1 = oof_probe_cate(Xdet, probe_resid, y0, y1, ite, groups)
    diag = {"auc_Y0": float(roc_auc_score(y0, p0)) if 0 < y0.mean() < 1 else float("nan"),
            "auc_Y1": float(roc_auc_score(y1, p1)) if 0 < y1.mean() < 1 else float("nan"),
            "auc_effect": (float(roc_auc_score((ite != 0).astype(int), np.abs(tauT)))
                           if 0 < (ite != 0).mean() < 1 else float("nan")),
            "rho_S_ite": float(spearmanr(tauS, ite).correlation),
            "rho_T_ite": float(spearmanr(tauT, ite).correlation)}
    yc, yk = off.outcome_vectors(events)
    haz, _ = off.oof_hazard(events, ALL)
    S = {"random": np.zeros(len(events)), "hazard (which fails)": haz,
         "semantic-CATE": sem_tau, "CATE (which helps)": tauT, "oracle": ite}
    rng = np.random.default_rng(0)
    curves = {k: e13.curve(s, yc, yk, rng) for k, s in S.items()}
    deltas = e13.bootstrap_deltas(events, S, yc, yk)         # uses S["CATE (which helps)"] vs random
    best_rho = max(diag["rho_S_ite"], diag["rho_T_ite"])
    predictive = (diag["auc_effect"] > 0.65) or (best_rho > 0.20)
    decision = deltas["auc_cate_rand"]["lo"] > 0
    return {"label": label, "n": len(events), "n_questions": len(set(groups)),
            "diagnostics": diag, "best_rho": float(best_rho),
            "curves": {k: [float(x) for x in v] for k, v in curves.items()}, "deltas": deltas,
            "gate": {"predictive_pass": bool(predictive), "decision_pass": bool(decision),
                     "PASS": bool(decision)}}   # PRIMARY (decision) is binding


def main():
    if not os.path.exists(PROBE_CACHE):
        print(f"missing {PROBE_CACHE} -- run experiments/e15_active_probe.py first"); return
    events = off.load_events()
    probe = pickle.load(open(PROBE_CACHE, "rb"))
    need = [f"{e['bench']}:{e['qid']}:{e['seed']}:{a}" for e in events for a in e["names"]]
    have = sum(1 for k in need if probe.get(k) not in (None, "__ERROR__"))
    print(f"probe coverage: {have}/{len(need)}")
    if have < 0.95 * len(need):
        print("probe cache incomplete (<95%) -- finish generation first"); return

    print("building probe features (+ flaw embeddings on gpu1)...")
    pscal_all, presid_all = build_probe_features(events, probe)
    print("computing E14 semantic-CATE baseline for comparison...")
    sem = e14.build_semantic(events)
    semtau_all = {}
    for lab, idx in [("pooled", list(range(len(events)))),
                     ("gsm8k", [i for i, e in enumerate(events) if e["bench"] == "gsm8k"]),
                     ("bbh", [i for i, e in enumerate(events) if e["bench"] == "bbh"])]:
        sc, mr, y0, y1, ite, gr = (a[idx] for a in sem)
        _, tT, _, _ = e14.oof_semantic(sc, mr, y0, y1, ite, gr)
        semtau_all[lab] = tT

    res = {}
    for lab, idx in [("pooled", list(range(len(events)))),
                     ("gsm8k", [i for i, e in enumerate(events) if e["bench"] == "gsm8k"]),
                     ("bbh", [i for i, e in enumerate(events) if e["bench"] == "bbh"])]:
        evs = [events[i] for i in idx]
        res[lab] = run_scope(evs, lab, pscal_all[idx], presid_all[idx], semtau_all[lab])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    for s in ("pooled", "gsm8k", "bbh"):
        r = res[s]; g = r["diagnostics"]; d = r["deltas"]; gt = r["gate"]
        print(f"\n=== {s} (n={r['n']}, {r['n_questions']} q) [PROBE-CATE] ===")
        print(f"  AUC(Y0={g['auc_Y0']:.2f}) AUC(Y1={g['auc_Y1']:.2f}) AUC(effect={g['auc_effect']:.2f}) "
              f"| rho(τ,ITE) S={g['rho_S_ite']:+.2f} T={g['rho_T_ite']:+.2f}")
        for bud in BUDGETS:
            cr = d[f"cate_rand@{bud}"]
            print(f"  @{int(bud*100)}%  Probe−random {cr['point']:+.3f} [{cr['lo']:+.3f},{cr['hi']:+.3f}]"
                  f"{'  *' if cr['lo']>0 else ''}")
        a = d["auc_cate_rand"]
        print(f"  AUC[20-40] Probe−random {a['point']:+.3f} [{a['lo']:+.3f},{a['hi']:+.3f}]"
              f"{'  *' if a['lo']>0 else ''}   GATE {'PASS' if gt['PASS'] else 'FAIL'} "
              f"(predictive {gt['predictive_pass']})")
    print(f"\n########## E15 PREREGISTERED VERDICT (pooled): "
          f"{'PASS -> proceed to E16' if res['pooled']['gate']['PASS'] else 'FAIL -> freeze E13/E14'} ##########")

    # figure
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#C7CFD8",
                         "legend.frameon": False})
    order = [("random", GREY, "-"), ("hazard (which fails)", PURPLE, "-"), ("semantic-CATE", TEAL, "-"),
             ("CATE (which helps)", BLUE, "-"), ("oracle", GOLD, "--")]
    lab = {"CATE (which helps)": "Probe-CATE (active)", "hazard (which fails)": "hazard"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharex=True)
    for ax, s in zip(axes, ("pooled", "gsm8k", "bbh")):
        r = res[s]
        for name, col, ls in order:
            ax.plot(CURVE, r["curves"][name], color=col, ls=ls,
                    lw=2.6 if name.startswith("CATE") else 1.7, marker="o", ms=3,
                    label=lab.get(name, name))
        ax.axvspan(0.20, 0.40, color="#DD8452", alpha=0.07, lw=0)
        ax.set_title(f"{s} (n={r['n']}) — gate {'PASS' if r['gate']['PASS'] else 'FAIL'}", fontsize=10)
        ax.set_xlabel("Intervention budget")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(True, alpha=0.28)
    axes[0].set_ylabel("Debate accuracy (policy value)")
    axes[0].legend(loc="lower right", fontsize=7.8)
    fig.suptitle("E15 — Active Diagnostic Probe: does acquiring new information (self-diagnosis) "
                 "make intervention benefit predictable?", fontsize=10.6, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"\n[OK] {OUT}\n[OK] {FIG}")


if __name__ == "__main__":
    main()
