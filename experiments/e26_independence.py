"""
E26 — Consensus is not epistemically homogeneous (ZERO-GPU).

Headline scientific claim (frozen):
    At fixed TOTAL support, INDEPENDENTLY-generated agreement carries substantially more
    correctness signal than SOCIALLY-INHERITED agreement.

This is the partial effect of round-0 independent support AFTER controlling total support — the
control is essential, because n_independent and n_support are ~0.83 correlated, so without it the
effect would merely restate E22 (marginal provenance -> correctness). Two equivalent parameter/isations:
    Model A:  logit P(Y=1) = b0 + b_ind*n_independent + b_sup*n_support        -> test b_ind>0
    Model B:  logit P(Y=1) = b0 + b_ind*n_independent + b_inh*n_inherited      (n_inherited=n_support-n_ind)
    identity: b_ind(A) = b_ind(B) - b_inh(B)   (per-vote value of independent MINUS inherited)
    Joint:    + n_persist  (decides whether "persistent" earns the headline alongside "independent")

NOTE on the joint model: {n_support, n_independent, n_inherited} are linearly dependent
(n_inherited = n_support - n_independent), so the identifiable basis is {n_independent, n_inherited,
n_persist}; first_round is dropped (E-diagnostic: redundant, beta n.s. once support+origin are known).

Candidate builder + correctness are reused verbatim from E22 (e22.candidates_for) across the full
5-ensemble x 8-benchmark grid (MATH removed). Inference is a near-unpenalised logistic (sklearn,
C=1e6, robust to the n_ind=3 -> P~1 near-separation); all CIs are clustered nonparametric bootstrap
on (config|bench|qid). ZERO-GPU. Deterministic (fixed seeds).

Outputs (paper-required):
  1. matched-stratum table  P(correct) by n_support x n_independent
  2. clustered-bootstrap coefficient table (Model A / B / joint / FE-adjusted)
  3. per-benchmark & per-config robustness of b_ind
  4. one clean figure -> results/figures/e26_matched_stratum.png
  + incremental AUC (why the downstream selector is only regime-bound) and actionable operating points.

  CUDA_VISIBLE_DEVICES="" python experiments/e26_independence.py
"""
import os, sys, glob, json
from collections import defaultdict
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import experiments.e22_provenance as e22

OUT = "results/tables/e26_independence.json"
FIG = "results/figures/e26_matched_stratum.png"
CORE = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa"]
NEW = ["aqua", "mmlu", "arc"]
BENCHES = CORE + NEW
CONFIGS = ["E_orig", "E_7b", "E_mixed", "E_strong", "E_n5"]
B_POOL, B_CELL = 1000, 600
MIN_STRATUM = 25          # min candidates to display a matched-stratum cell


def logdir(cfg, b):
    if cfg == "E_orig":
        return "data/debate_logs" if (b in CORE or b == "aqua") else "data/debate_logs/gen/E_orig3b"
    return f"data/debate_logs/gen/{cfg}"


def build_candidates():
    rows = []
    for cfg in CONFIGS:
        for b in BENCHES:
            ld = logdir(cfg, b)
            if not glob.glob(f"{ld}/{b}_seed*.jsonl"):
                continue
            for r in e22.candidates_for(b, logdir=ld):
                r["config"] = cfg
                r["cluster"] = f"{cfg}|{b}|{r['qid']}"
                r["n_inherited"] = r["n_support"] - r["n_independent"]
                rows.append(r)
    return rows


def _fit(X, y):
    if len(set(y)) < 2:
        return None
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs").fit(X, y).coef_[0]


def coefs_ci(rows, feats, B, seed=0):
    """Full-model fit + clustered bootstrap CI for EVERY coefficient in `feats`."""
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["correct"] for r in rows], int)
    full = _fit(X, y)
    if full is None:
        return None
    cl = defaultdict(list)
    for i, r in enumerate(rows):
        cl[r["cluster"]].append(i)
    keys = list(cl.values())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        pick = np.concatenate([keys[j] for j in rng.integers(0, len(keys), len(keys))])
        b = _fit(X[pick], y[pick])
        if b is not None:
            boot.append(b)
    boot = np.array(boot)
    out = {}
    for k, f in enumerate(feats):
        col = boot[:, k]
        out[f] = {"beta": float(full[k]), "lo": float(np.percentile(col, 2.5)),
                  "hi": float(np.percentile(col, 97.5)), "p_le0": float(np.mean(col <= 0))}
    return out


def fe_beta_ind(rows, B, seed=1):
    benches_u = sorted(set(r["bench"] for r in rows))
    cfgs_u = sorted(set(r["config"] for r in rows))
    def design(sub):
        X = []
        for r in sub:
            row = [r["n_independent"], r["n_support"]]
            row += [1.0 if r["bench"] == bb else 0.0 for bb in benches_u[1:]]
            row += [1.0 if r["config"] == cc else 0.0 for cc in cfgs_u[1:]]
            X.append(row)
        return np.array(X, float), np.array([r["correct"] for r in sub], int)
    X, y = design(rows)
    full = _fit(X, y)
    cl = defaultdict(list)
    for i, r in enumerate(rows):
        cl[r["cluster"]].append(i)
    keys = list(cl.values())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        pick = np.concatenate([keys[j] for j in rng.integers(0, len(keys), len(keys))])
        b = _fit(X[pick], y[pick])
        if b is not None:
            bs.append(b[0])
    bs = np.array(bs)
    return {"beta": float(full[0]), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "p_le0": float(np.mean(bs <= 0))}


def matched_stratum(rows):
    cell = defaultdict(list)
    for r in rows:
        cell[(r["n_support"], r["n_independent"])].append(r["correct"])
    out = {}
    for (s, i), c in cell.items():
        if len(c) >= MIN_STRATUM:
            out.setdefault(s, {})[i] = [float(np.mean(c)), len(c)]
    return out


def incremental_auc(rows):
    y = np.array([r["correct"] for r in rows], int)
    def auc(feats):
        X = np.array([[r[f] for f in feats] for r in rows], float)
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        return float(roc_auc_score(y, m.decision_function(X)))
    a_sup, a_ind, a_both = auc(["n_support"]), auc(["n_independent"]), auc(["n_support", "n_independent"])
    return {"n_support": a_sup, "n_independent": a_ind, "both": a_both, "delta_over_support": a_both - a_sup}


def operating_points(rows):
    y = np.array([r["correct"] for r in rows], int)
    ni = np.array([r["n_independent"] for r in rows], int)
    op = {}
    for v in sorted(set(ni)):
        m = ni == v
        if m.sum() >= 30:
            op[f"n_ind={v}"] = {"p_correct": float(y[m].mean()), "coverage": float(m.mean()), "n": int(m.sum())}
    op["rule_trust_ge2"] = {"p_correct": float(y[ni >= 2].mean()), "coverage": float((ni >= 2).mean())}
    op["rule_reject_eq0"] = {"p_correct": float(y[ni == 0].mean()), "coverage": float((ni == 0).mean())}
    return op


def make_figure(strata, path):
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    # keep only strata that form an actual line (>=2 distinct n_ind points); drops sparse high-support singletons
    svals = [s for s in sorted(strata.keys()) if len(strata[s]) >= 2]
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    cmap = plt.cm.viridis(np.linspace(0.06, 0.86, len(svals)))
    for col, s in zip(cmap, svals):
        pts = sorted(strata[s].items())
        xs = [i for i, _ in pts]
        ys = [pn[0] for _, pn in pts]
        ns = [pn[1] for _, pn in pts]
        ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=2, label=f"{s}")
        ax.scatter(xs, ys, s=[max(28, min(300, n / 2.2)) for n in ns], color=col,
                   zorder=3, alpha=.9, edgecolors="white", linewidths=.6)
    ax.set_xlabel("independent support  $n_{\\mathrm{independent}}$  (round-0 origin)")
    ax.set_ylabel("P(correct)")
    ax.set_title("Consensus is not epistemically homogeneous\n"
                 "at fixed total support, independent origin raises P(correct)", fontsize=12.5, loc="left")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.25)
    ax.legend(title="total support $n_{\\mathrm{support}}$", loc="lower right", frameon=False,
              ncol=3, fontsize=9, title_fontsize=9, columnspacing=1.1, handlelength=1.4)
    fig.text(0.012, 0.005, "each line = one total-support stratum; reading left→right swaps inherited "
             "votes for independent ones (marker area ∝ n).", fontsize=7.6, color="#666")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run():
    rows = build_candidates()
    y = np.array([r["correct"] for r in rows], int)
    ni = np.array([r["n_independent"] for r in rows], float)
    ns = np.array([r["n_support"] for r in rows], float)
    corr = float(np.corrcoef(ni, ns)[0, 1])
    print(f"candidates={len(rows)}  configs={CONFIGS}  benches={BENCHES}")
    print(f"base P(correct)={y.mean():.3f}   corr(n_ind,n_support)={corr:.3f}\n")

    mA = coefs_ci(rows, ["n_independent", "n_support"], B_POOL, seed=0)
    mB = coefs_ci(rows, ["n_independent", "n_inherited"], B_POOL, seed=2)
    mJ = coefs_ci(rows, ["n_independent", "n_inherited", "n_persist"], B_POOL, seed=4)
    mFE = fe_beta_ind(rows, B_POOL, seed=1)
    strata = matched_stratum(rows)
    inc = incremental_auc(rows)
    ops = operating_points(rows)

    per_bench = {b: coefs_ci([r for r in rows if r["bench"] == b], ["n_independent", "n_support"], B_CELL, seed=10)
                 for b in BENCHES}
    per_config = {c: coefs_ci([r for r in rows if r["config"] == c], ["n_independent", "n_support"], B_CELL, seed=20)
                  for c in CONFIGS}

    def fmt(d):
        return f"{d['beta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]{'*' if d['lo'] > 0 else ' '} p(<=0)={d['p_le0']:.3f}"

    print("=" * 72 + "\n1+2. COEFFICIENT TABLE (clustered bootstrap, per-vote log-odds)\n" + "=" * 72)
    print(f"  Model A  b_independent | ctrl n_support     = {fmt(mA['n_independent'])}")
    print(f"  Model B  b_independent (per indep vote)     = {fmt(mB['n_independent'])}")
    print(f"  Model B  b_inherited   (per inherited vote) = {fmt(mB['n_inherited'])}")
    ratio = mB['n_independent']['beta'] / mB['n_inherited']['beta']
    print(f"           per-vote value ratio indep:inherit = {ratio:.2f}x")
    print(f"  Model A + bench&config FE                   = {fmt(mFE)}")
    print(f"  JOINT    b_independent                      = {fmt(mJ['n_independent'])}")
    print(f"  JOINT    b_inherited                        = {fmt(mJ['n_inherited'])}")
    print(f"  JOINT    b_persist                          = {fmt(mJ['n_persist'])}   <- decides 'persistent' in headline")

    print("\n" + "=" * 72 + "\n3. ROBUSTNESS — b_ind per benchmark / per config\n" + "=" * 72)
    for b in BENCHES:
        d = per_bench[b]
        print(f"  {b:14} {fmt(d['n_independent'])}")
    print("  " + "-" * 40)
    for c in CONFIGS:
        d = per_config[c]
        print(f"  {c:14} {fmt(d['n_independent'])}")

    print("\n" + "=" * 72 + "\n1. MATCHED-STRATUM  P(correct) by n_support x n_ind (n>=%d)\n" % MIN_STRATUM + "=" * 72)
    ivals = sorted({i for s in strata.values() for i in s})
    print("  n_sup \\ n_ind |" + "".join(f"{i:>12}" for i in ivals))
    for s in sorted(strata):
        cells = []
        for i in ivals:
            pn = strata[s].get(i)
            cells.append(f"{pn[0]:.2f}(n{pn[1]})" if pn else "-")
        print(f"  {s:>12}  |" + "".join(f"{c:>12}" for c in cells))

    print("\n" + "=" * 72 + "\nBRIDGE TO UTILITY — incremental AUC & operating points\n" + "=" * 72)
    print(f"  AUC n_support={inc['n_support']:.3f}  +n_independent={inc['both']:.3f}  "
          f"(Δ={inc['delta_over_support']:+.3f}) -> origin adds little to RANKING (collinear); "
          f"explains why the selector is regime-bound, not universal")
    print(f"  reject rule (n_ind==0): P(correct)={ops['rule_reject_eq0']['p_correct']:.3f} "
          f"covers {ops['rule_reject_eq0']['coverage']*100:.0f}%   |   "
          f"trust rule (n_ind>=2): P(correct)={ops['rule_trust_ge2']['p_correct']:.3f} "
          f"covers {ops['rule_trust_ge2']['coverage']*100:.0f}%")

    make_figure(strata, FIG)
    print(f"\n[fig] {FIG}")

    result = {
        "n_candidates": len(rows), "configs": CONFIGS, "benches": BENCHES,
        "base_rate": float(y.mean()), "corr_ind_support": corr,
        "model_A_ctrl_support": mA, "model_B_decomposition": mB,
        "per_vote_ratio_indep_over_inherited": ratio,
        "model_A_fixed_effects": {"n_independent": mFE},
        "model_joint_ind_inh_persist": mJ,
        "per_benchmark": {b: per_bench[b]["n_independent"] for b in BENCHES},
        "per_config": {c: per_config[c]["n_independent"] for c in CONFIGS},
        "matched_stratum": {str(s): {str(i): v for i, v in d.items()} for s, d in strata.items()},
        "incremental_auc": inc, "operating_points": ops, "figure": FIG,
        "_note": ("Headline: at fixed total support, independent agreement carries more correctness "
                  "signal than inherited agreement (Model A b_ind>0, CI-lo>0). Model B gives per-vote "
                  "values; JOINT decides whether persistence is a co-headline (b_persist CI). Incremental "
                  "AUC over n_support is small (collinearity) — this EXPLAINS the modest, regime-bound "
                  "downstream selector rather than contradicting the mechanism."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"[OK] {OUT}")
    return result


if __name__ == "__main__":
    run()
