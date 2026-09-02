"""
E14-B2 (PREREGISTERED, one shot) — does SEMANTIC round-0 information make the treatment
effect predictable? Changes ONLY the feature vector vs E13 (same g=0 gate, same ITE=Y1-Y0,
same S/T learners, same policy eval + gate). Embeddings from nomic-embed-text on gpu1.

Features are RESIDUALIZED (subtract per-question centroid) to kill topic/benchmark leakage
(E3b-v2 lesson), order-invariant over agents. Four groups:
  own-reasoning content : residual-norm stats + PCA(8) of event-mean residual embedding
  peer similarity       : residual pairwise cosine (mean/min/max/std)
  own-vs-peer gap       : cos(agent, mean-of-peers) residual (mean/min/max)
  raw reference         : raw pairwise cosine mean
PCA is fit WITHIN each training fold (no leakage). Residualization is per-question and
each question lives entirely in one fold, so it is leakage-free.

LOCKED GATE (paper/E14_benefit_features_prereg.md):
  predictive: AUC_effect > 0.65  OR  rho(tau_hat, ITE) > 0.20
  decision  : AUC[20-40] (benefit - random) cluster-bootstrap CI lower bound > 0
  PASS = both.  FAIL = stop benefit-controller; E13 is the contribution.

Output: results/tables/e14_benefit_semantic.json , results/figures/e14_benefit_semantic.png
"""
import os, sys, json, pickle, hashlib, urllib.request
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.e6_offline as off
import experiments.e13_benefit_aware as e13

OUT = "results/tables/e14_benefit_semantic.json"
FIG = "results/figures/e14_benefit_semantic.png"
NOMIC = "results/cache/nomic_embed.pkl"
EMB_URL = "http://127.0.0.1:11435/api/embeddings"   # gpu1-pinned
BUDGETS = e13.BUDGETS
CURVE = e13.CURVE
N_PCA = 8
INK = "#2B2B2B"
BLUE, ORANGE, GREEN, GREY, PURPLE, GOLD = (
    "#4C72B0", "#DD8452", "#55A868", "#8C8C8C", "#8172B3", "#C7A008")


# ------------------------------- embeddings -----------------------------------
def _embed_api(t):
    if not t or not t.strip():
        return np.zeros(768)
    req = urllib.request.Request(EMB_URL, headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": "nomic-embed-text", "prompt": t}).encode())
    with urllib.request.urlopen(req, timeout=60) as r:
        return np.asarray(json.load(r)["embedding"], float)


def embed_texts(texts):
    cache = pickle.load(open(NOMIC, "rb")) if os.path.exists(NOMIC) else {}
    out = {}; new = 0
    for t in texts:
        h = hashlib.sha1((t or "").encode("utf-8")).hexdigest()
        if h in cache:
            out[t] = np.asarray(cache[h], float); continue
        v = _embed_api(t); cache[h] = v; out[t] = v; new += 1
        if new % 50 == 0:
            pickle.dump(cache, open(NOMIC, "wb")); print(f"  embedded {new} new...")
    pickle.dump(cache, open(NOMIC, "wb"))
    print(f"  embeddings ready ({new} new, {len(texts)-new} reused)")
    return out


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


# --------------------- residualized semantic features -------------------------
def build_semantic(events):
    """Returns (scalars [n,K], mean_resid [n,768], y0, y1, ite, groups). Residualize per qid."""
    agents = events[0]["names"]
    texts = {(i, a): (e["trigger_state"][a]["reasoning"] or "")
             for i, e in enumerate(events) for a in e["names"]}
    emb_map = embed_texts(list(set(texts.values())))
    E = {(i, a): emb_map[t] for (i, a), t in texts.items()}

    # per-question centroid over all (agent, event) embeddings of that question
    byq = defaultdict(list)
    for i, e in enumerate(events):
        for a in e["names"]:
            byq[(e["bench"], e["qid"])].append(E[(i, a)])
    centroid = {q: np.mean(v, axis=0) for q, v in byq.items()}

    scalars, mean_resid = [], []
    for i, e in enumerate(events):
        c = centroid[(e["bench"], e["qid"])]
        emb = [E[(i, a)] for a in e["names"]]
        res = [v - c for v in emb]
        pairs = [(x, y) for x in range(len(res)) for y in range(x + 1, len(res))]
        sim = [_cos(res[x], res[y]) for x, y in pairs]
        raw = [_cos(emb[x], emb[y]) for x, y in pairs]
        gap = [_cos(res[a], np.mean([res[b] for b in range(len(res)) if b != a], axis=0))
               for a in range(len(res))]
        nrm = [float(np.linalg.norm(v)) for v in res]
        scalars.append([
            np.mean(sim), np.min(sim), np.max(sim), np.std(sim),
            np.mean(gap), np.min(gap), np.max(gap),
            np.mean(nrm), np.max(nrm), np.std(nrm),
            np.mean(raw)])
        mean_resid.append(np.mean(res, axis=0))
    y0 = np.array([int(bool(e["control"]["rec_maj"])) for e in events])
    y1 = np.array([int(bool(e["combined"]["rec_maj"])) for e in events])
    groups = np.array([e["qid"] for e in events])
    return (np.array(scalars, float), np.array(mean_resid, float),
            y0, y1, (y1 - y0).astype(float), groups)


def _feat_fold(scalars, mresid, tr, te):
    """Within-fold features: scalars + PCA(N_PCA) of residual embeddings fit on train."""
    p = PCA(n_components=min(N_PCA, mresid.shape[1], len(tr) - 1))
    ztr = p.fit_transform(mresid[tr]); zte = p.transform(mresid[te])
    Xtr = np.hstack([scalars[tr], ztr]); Xte = np.hstack([scalars[te], zte])
    return Xtr, Xte


def oof_semantic(scalars, mresid, y0, y1, ite, groups, seed=0):
    ns = min(5, np.unique(groups).size)
    tauS = np.zeros(len(ite)); p0 = np.full(len(ite), y0.mean(), float)
    p1 = np.full(len(ite), y1.mean(), float)
    if ns >= 2:
        for tr, te in GroupKFold(ns).split(scalars, ite, groups):
            Xtr, Xte = _feat_fold(scalars, mresid, tr, te)
            rr = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.06,
                                               l2_regularization=1.0, random_state=seed)
            rr.fit(Xtr, ite[tr]); tauS[te] = rr.predict(Xte)
            for y, pv in ((y1, p1), (y0, p0)):
                if 0 < y[tr].mean() < 1:
                    cc = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                                        l2_regularization=1.0, random_state=seed)
                    cc.fit(Xtr, y[tr]); pv[te] = cc.predict_proba(Xte)[:, 1]
    return tauS, (p1 - p0), p0, p1


def run_scope(events, label, sem):
    scalars, mresid, y0, y1, ite, groups = sem
    yc, yk = off.outcome_vectors(events)
    never = float(yc.mean())
    tauS, tauT, p0, p1 = oof_semantic(scalars, mresid, y0, y1, ite, groups)
    diag = {
        "auc_Y0": float(roc_auc_score(y0, p0)) if 0 < y0.mean() < 1 else float("nan"),
        "auc_Y1": float(roc_auc_score(y1, p1)) if 0 < y1.mean() < 1 else float("nan"),
        "auc_effect": (float(roc_auc_score((ite != 0).astype(int), np.abs(tauT)))
                       if 0 < (ite != 0).mean() < 1 else float("nan")),
        "rho_S_ite": float(spearmanr(tauS, ite).correlation),
        "rho_T_ite": float(spearmanr(tauT, ite).correlation)}
    haz, _ = off.oof_hazard(events, ("f1_", "f2_", "f3_", "f4_"))
    S = {"random": np.zeros(len(events)),
         "hazard (which fails)": haz,
         "CATE (which helps)": tauT,          # semantic T-learner is the benefit-aware policy
         "oracle (true ITE)": ite}
    rng = np.random.default_rng(0)
    curves = {k: e13.curve(s, yc, yk, rng) for k, s in S.items()}
    deltas = e13.bootstrap_deltas(events, S, yc, yk)
    orc = off.acc_at_budget(ite, yc, yk, 0.40, rng)[0] - never
    cat = off.acc_at_budget(tauT, yc, yk, 0.40, rng)[0] - off.acc_at_budget(np.zeros(len(events)), yc, yk, 0.40, rng)[0]
    return {"label": label, "n": len(events), "n_questions": len(set(groups)),
            "never": never, "diagnostics": diag,
            "curves": {k: [float(x) for x in v] for k, v in curves.items()},
            "deltas": deltas, "oracle_gain@40": float(orc), "cate_edge@40": float(cat),
            "best_rho": float(max(diag["rho_S_ite"], diag["rho_T_ite"]))}


def gate(r):
    predictive = (r["diagnostics"]["auc_effect"] > 0.65) or (r["best_rho"] > 0.20)
    decision = r["deltas"]["auc_cate_rand"]["lo"] > 0
    return {"predictive_pass": bool(predictive), "decision_pass": bool(decision),
            "PASS": bool(predictive and decision)}


def main():
    events = off.load_events()
    print("building semantic features (embeddings on gpu1 :11435)...")
    sem_all = build_semantic(events)
    scopes = {"pooled": (events, sem_all)}
    # per-benchmark: slice the precomputed arrays by bench
    for b in ("gsm8k", "bbh"):
        idx = [i for i, e in enumerate(events) if e["bench"] == b]
        evs = [events[i] for i in idx]
        sem_b = tuple(a[idx] if hasattr(a, "__len__") and getattr(a, "ndim", 1) >= 1 and len(a) == len(events)
                      else a for a in sem_all)
        scopes[b] = (evs, sem_b)

    res = {}
    for s, (evs, sem) in scopes.items():
        res[s] = run_scope(evs, s, sem)
        res[s]["gate"] = gate(res[s])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    for s in ("pooled", "gsm8k", "bbh"):
        r = res[s]; d = r["deltas"]; g = r["diagnostics"]; gt = r["gate"]
        print(f"\n=== {s}  (n={r['n']}, {r['n_questions']} q) [SEMANTIC features] ===")
        print(f"  predictability: AUC(Y0={g['auc_Y0']:.2f}) AUC(Y1={g['auc_Y1']:.2f}) "
              f"AUC(effect={g['auc_effect']:.2f}) | rho(τ,ITE) S={g['rho_S_ite']:+.2f} T={g['rho_T_ite']:+.2f}")
        for bud in BUDGETS:
            cr = d[f"cate_rand@{bud}"]
            print(f"  @{int(bud*100)}%  CATE−random {cr['point']:+.3f} [{cr['lo']:+.3f},{cr['hi']:+.3f}]"
                  f"{'  *' if cr['lo']>0 else ''}")
        a = d["auc_cate_rand"]
        print(f"  AUC[20-40] CATE−random {a['point']:+.3f} [{a['lo']:+.3f},{a['hi']:+.3f}]"
              f"{'  *' if a['lo']>0 else ''}")
        print(f"  GATE: predictive={gt['predictive_pass']} decision={gt['decision_pass']} "
              f"-> {'PASS' if gt['PASS'] else 'FAIL'}")

    print(f"\n########## PREREGISTERED VERDICT (pooled): "
          f"{'PASS' if res['pooled']['gate']['PASS'] else 'FAIL'} ##########")

    # ------------------------------- figure --------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#C7CFD8", "legend.frameon": False})
    order = ["random", "hazard (which fails)", "CATE (which helps)", "oracle (true ITE)"]
    col = {"random": GREY, "hazard (which fails)": PURPLE,
           "CATE (which helps)": BLUE, "oracle (true ITE)": GOLD}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharex=True)
    for ax, s in zip(axes, ("pooled", "gsm8k", "bbh")):
        r = res[s]
        for name in order:
            lw = 2.6 if name.startswith("CATE") else 1.7
            ls = "--" if name.startswith("oracle") else "-"
            lab = "CATE-semantic (benefit)" if name.startswith("CATE") else name
            ax.plot(CURVE, r["curves"][name], color=col[name], lw=lw, ls=ls, marker="o", ms=3, label=lab)
        ax.axvspan(0.20, 0.40, color=ORANGE, alpha=0.07, lw=0)
        v = "PASS" if r["gate"]["PASS"] else "FAIL"
        ax.set_title(f"{s} (n={r['n']}) — gate {v}", fontsize=10)
        ax.set_xlabel("Intervention budget")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(True, alpha=0.28)
    axes[0].set_ylabel("Debate accuracy (policy value)")
    axes[0].legend(loc="lower right", fontsize=7.8)
    fig.suptitle("E14-B2 — Benefit-aware controller with RESIDUALIZED SEMANTIC round-0 features "
                 "(preregistered one shot). Does richer information make benefit predictable?",
                 fontsize=10.6, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"\n[OK] {OUT}\n[OK] {FIG}")


if __name__ == "__main__":
    main()
