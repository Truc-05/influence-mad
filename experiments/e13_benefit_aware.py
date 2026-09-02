"""
E13 — Benefit-Aware controller: uplift / CATE targeting (ZERO-GPU).

Thesis: FAILURE prediction != INTERVENTION-BENEFIT prediction. The E3a hazard answers
"which debate fails?" (P[Y0=0]); the controller actually needs "which debate does the
safeguard HELP?" (the treatment effect). BBH is the proof they differ: hazard is
detectable there yet targeting it does not beat random.

Lucky structure: the E6 precompute stored BOTH arms for every debate, so we observe both
potential outcomes per unit and the individual treatment effect is known:
    Y0 = control.rec_maj,  Y1 = combined.rec_maj,  ITE = Y1 - Y0 in {-1, 0, +1}.
Any policy that intervenes on set S has accuracy  never + (1/n) * sum_{i in S} ITE_i,
so the OPTIMAL policy targets the largest ITE, and targeting true ITE is an oracle
ceiling. We learn CATE  tau_hat(x) = E[ITE | X=x]  from round-0 features by
group-cross-fitting over questions, and compare policies at matched budget:
    random | disagreement heuristic | hazard (E3a, "which fails") | CATE ("which helps")
    | oracle (true ITE, ceiling).
Primary: Delta Acc(CATE - random) and (CATE - hazard) with cluster-bootstrap over qids;
policy-value curves over budgets 20-60%. Reported pooled and per-benchmark (BBH is the
decisive case).

Output: results/tables/e13_benefit_aware.json , results/figures/e13_benefit_aware.png
"""
import os, sys, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.e6_offline as off

OUT = "results/tables/e13_benefit_aware.json"
FIG = "results/figures/e13_benefit_aware.png"
BUDGETS = [0.20, 0.30, 0.40, 0.50, 0.60]
CURVE = np.round(np.arange(0.05, 0.801, 0.05), 3)
ALL = ("f1_", "f2_", "f3_", "f4_")
B = 4000
INK = "#2B2B2B"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, GOLD = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C", "#8172B3", "#C7A008")


def _XY(events):
    keys = off.feature_keys(events, ALL)
    X = np.array([[e["feat"][k] for k in keys] for e in events])
    y0 = np.array([int(bool(e["control"]["rec_maj"])) for e in events])
    y1 = np.array([int(bool(e["combined"]["rec_maj"])) for e in events])
    groups = np.array([e["qid"] for e in events])
    return X, y0, y1, groups


def _oof_reg(X, y, groups, seed=0):
    ns = min(5, np.unique(groups).size)
    pred = np.zeros(len(y))
    if ns >= 2:
        for tr, te in GroupKFold(ns).split(X, y, groups):
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.06,
                                              l2_regularization=1.0, random_state=seed)
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def _oof_clf(X, y, groups, seed=0):
    ns = min(5, np.unique(groups).size, int(y.sum()), int((1 - y).sum()))
    pred = np.full(len(y), y.mean(), float)
    if ns >= 2:
        for tr, te in GroupKFold(ns).split(X, y, groups):
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                               l2_regularization=1.0, random_state=seed)
            m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def scores_for(events):
    X, y0, y1, groups = _XY(events)
    ite = (y1 - y0).astype(float)
    tau_S = _oof_reg(X, ite, groups)                       # S-learner: regress ITE directly
    p1, p0 = _oof_clf(X, y1, groups), _oof_clf(X, y0, groups)
    tau_T = p1 - p0                                        # T-learner: two models (standard uplift)
    haz, _ = off.oof_hazard(events, ALL)
    diag = {"spearman_S_ite": float(spearmanr(tau_S, ite).correlation),
            "spearman_T_ite": float(spearmanr(tau_T, ite).correlation),
            "auc_predict_Y0": float(roc_auc_score(y0, p0)) if 0 < y0.mean() < 1 else float("nan"),
            "auc_predict_Y1": float(roc_auc_score(y1, p1)) if 0 < y1.mean() < 1 else float("nan"),
            "auc_predict_effect": (float(roc_auc_score((ite != 0).astype(int),
                                   np.abs(tau_T))) if 0 < (ite != 0).mean() < 1 else float("nan"))}
    scores = {
        "random": np.zeros(len(events)),
        "disagreement": np.array([e["feat"]["f1_disagree_ratio"] for e in events], float),
        "hazard (which fails)": haz,
        "CATE (which helps)": tau_T,                       # T-learner is the headline CATE
        "oracle (true ITE)": ite,
    }
    return scores, tau_T, ite, diag


def _pv(score, yc, yk, budget, rng):
    """Policy value: one random-tie-break top-k, mean(Y1 if fire else Y0). Fast (single draw)."""
    n = len(score); k = int(round(budget * n))
    if k <= 0:
        return float(yc.mean())
    if k >= n:
        return float(yk.mean())
    order = np.lexsort((rng.random(n), -score))
    fire = np.zeros(n, bool); fire[order[:k]] = True
    return float(np.mean(np.where(fire, yk, yc)))


def curve(score, yc, yk, rng, R=200):
    return [float(np.mean([_pv(score, yc, yk, b, rng) for _ in range(R)])) for b in CURVE]


def bootstrap_deltas(events, S, yc, yk):
    """Cluster-bootstrap Delta Acc(CATE - random) and (CATE - hazard) at BUDGETS + AUC.
    Single tie-break draw per bootstrap replicate (the bootstrap supplies the variance)."""
    clusters = defaultdict(list)
    for i, e in enumerate(events):
        clusters[e["qid"]].append(i)
    ck = list(clusters.values())
    rng = np.random.default_rng(0)
    cate, rand, haz = S["CATE (which helps)"], S["random"], S["hazard (which fails)"]
    grid = np.round(np.arange(0.20, 0.601, 0.05), 3)

    def deltas(idx):
        c, r, h, a, b = cate[idx], rand[idx], haz[idx], yc[idx], yk[idx]
        out = {}
        for bud in BUDGETS:
            out[f"cate_rand@{bud}"] = _pv(c, a, b, bud, rng) - _pv(r, a, b, bud, rng)
            out[f"cate_haz@{bud}"] = _pv(c, a, b, bud, rng) - _pv(h, a, b, bud, rng)
        out["auc_cate_rand"] = float(np.mean([_pv(c, a, b, g, rng) - _pv(r, a, b, g, rng)
                                              for g in grid]))
        return out

    full = np.arange(len(events))
    acc = None
    for _ in range(200):
        d = deltas(full)
        if acc is None:
            acc = {k: [] for k in d}
        for k in acc:
            acc[k].append(d[k])
    point = {k: float(np.mean(v)) for k, v in acc.items()}
    draws = {k: np.empty(B) for k in point}
    for bo in range(B):
        idx = np.concatenate([ck[j] for j in rng.integers(0, len(ck), len(ck))])
        d = deltas(idx)
        for k in point:
            draws[k][bo] = d[k]
    return {k: {"point": point[k], "lo": float(np.percentile(draws[k], 2.5)),
                "hi": float(np.percentile(draws[k], 97.5))} for k in point}


def run_scope(events, label):
    S, tau, ite, diag = scores_for(events)
    yc, yk = off.outcome_vectors(events)
    never = float(yc.mean())
    rng = np.random.default_rng(0)
    curves = {name: curve(s, yc, yk, rng) for name, s in S.items()}
    deltas = bootstrap_deltas(events, S, yc, yk)
    # fraction of oracle gain captured by CATE at 40%
    orc40 = off.acc_at_budget(ite, yc, yk, 0.40, rng)[0] - never
    cate40 = off.acc_at_budget(tau, yc, yk, 0.40, rng)[0] - never
    rand40 = off.acc_at_budget(np.zeros(len(events)), yc, yk, 0.40, rng)[0] - never
    frac = (cate40 - rand40) / (orc40 - rand40) if (orc40 - rand40) > 1e-9 else float("nan")
    return {"label": label, "n": len(events),
            "n_questions": len({e["qid"] for e in events}),
            "never": never, "diagnostics": diag,
            "curves": {k: [float(x) for x in v] for k, v in curves.items()},
            "deltas": deltas,
            "oracle_gain@40": float(orc40), "cate_gain@40": float(cate40 - rand40),
            "frac_oracle_captured@40": float(frac)}


def main():
    events = off.load_events()
    scopes = {"pooled": events,
              "gsm8k": [e for e in events if e["bench"] == "gsm8k"],
              "bbh": [e for e in events if e["bench"] == "bbh"]}
    res = {s: run_scope(evs, s) for s, evs in scopes.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    for s in ("pooled", "gsm8k", "bbh"):
        r = res[s]; d = r["deltas"]; g = r["diagnostics"]
        print(f"\n=== {s}  (n={r['n']}, {r['n_questions']} q, never={r['never']:.3f}) ===")
        print(f"  predictability: AUC(Y0={g['auc_predict_Y0']:.2f}) AUC(Y1={g['auc_predict_Y1']:.2f}) "
              f"AUC(effect≠0={g['auc_predict_effect']:.2f}) | "
              f"spearman(τ,ITE): S={g['spearman_S_ite']:+.2f} T={g['spearman_T_ite']:+.2f}")
        for bud in BUDGETS:
            cr, ch = d[f"cate_rand@{bud}"], d[f"cate_haz@{bud}"]
            sg = "*" if cr["lo"] > 0 else " "
            print(f"  @{int(bud*100)}%  CATE−random {cr['point']:+.3f} "
                  f"[{cr['lo']:+.3f},{cr['hi']:+.3f}]{sg}   CATE−hazard {ch['point']:+.3f} "
                  f"[{ch['lo']:+.3f},{ch['hi']:+.3f}]")
        a = d["auc_cate_rand"]
        print(f"  AUC[20-40] CATE−random {a['point']:+.3f} [{a['lo']:+.3f},{a['hi']:+.3f}]"
              f"{'  *' if a['lo']>0 else ''}")
        print(f"  CATE captures {r['frac_oracle_captured@40']:.0%} of the oracle gain @40% "
              f"(oracle +{r['oracle_gain@40']:.3f}, CATE +{r['cate_gain@40']:.3f})")

    # ------------------------------- figure --------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#C7CFD8", "legend.frameon": False})
    order = ["random", "disagreement", "hazard (which fails)", "CATE (which helps)", "oracle (true ITE)"]
    col = {"random": GREY, "disagreement": ORANGE, "hazard (which fails)": PURPLE,
           "CATE (which helps)": BLUE, "oracle (true ITE)": GOLD}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharex=True)
    for ax, s in zip(axes, ("pooled", "gsm8k", "bbh")):
        r = res[s]
        for name in order:
            lw = 2.6 if name.startswith("CATE") else (1.6 if name in ("random", "oracle (true ITE)") else 1.9)
            ls = "--" if name == "oracle (true ITE)" else "-"
            ax.plot(CURVE, r["curves"][name], color=col[name], lw=lw, ls=ls,
                    marker="o", ms=3, label=name)
        ax.axvspan(0.20, 0.40, color=ORANGE, alpha=0.07, lw=0)
        ax.set_title(f"{s}  (n={r['n']})", fontsize=10.5)
        ax.set_xlabel("Intervention budget")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(True, alpha=0.28)
    axes[0].set_ylabel("Debate accuracy (policy value)")
    axes[0].legend(loc="lower right", fontsize=7.6)
    fig.suptitle("E13 — Benefit-Aware controller: targeting treatment effect (CATE) vs "
                 "targeting failure (hazard). Failure prediction ≠ intervention benefit.",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"\n[OK] {OUT}\n[OK] {FIG}")


if __name__ == "__main__":
    main()
