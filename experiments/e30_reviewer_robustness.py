"""
E30 — Reviewer-robustness battery (ZERO-GPU). Addresses KBS reviewer points that are computable from
the frozen logs / candidate table:

  (A) COST accounting: LLM calls for round-0-only vs full debate, and the accuracy of round-0 majority
      vs deployed final majority (why "just stop at round 0" is the honest practical takeaway).
  (B) COLLINEARITY stability: VIF for the H3 feature sets + ridge-path stability of b_ind (is the
      partial coefficient an artifact of r=0.83 collinearity?).
  (C) DROP-PERSISTENCE ablation: b_ind in {ind,adopt,per} vs {ind,adopt} vs {ind,sup}.
  (D) DATA COUNTS: questions / seeds / candidates per (ensemble, benchmark) — one reproducibility table.
  (E) POOL fragmentation: mean distinct hypotheses per question per benchmark (explains TruthfulQA=6.31).
  (F) CORRELATED ERROR: at round 0, observed all-agents-wrong rate vs the independence prediction
      (product of marginal error rates) — tests the "interaction-independent, not statistically
      independent" caveat honestly.
  (G) EXTINCTION vs BASELINE ACCURACY: is dCR just a function of how weak the base system is?

  CUDA_VISIBLE_DEVICES="" python experiments/e30_reviewer_robustness.py
"""
import os, sys, glob, json
from collections import defaultdict
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from sklearn.linear_model import LogisticRegression, LinearRegression
from src.utils.io_utils import read_jsonl
import experiments.e3_router as e3
import experiments.e26_independence as e26
import experiments.e28_h3_robust as e28

OUT = "results/tables/e30_reviewer_robustness.json"
CORE = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa"]


def vif(rows, feats):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    out = {}
    for j, f in enumerate(feats):
        others = [k for k in range(len(feats)) if k != j]
        r2 = LinearRegression().fit(X[:, others], X[:, j]).score(X[:, others], X[:, j])
        out[f] = float(1.0 / max(1e-9, 1.0 - r2))
    return out


def ridge_path(rows, feats, Cs=(1e6, 10, 1, 0.1, 0.01)):
    Xs = np.array([[r[f] for f in feats] for r in rows], float)
    Xs = (Xs - Xs.mean(0)) / Xs.std(0)            # standardized -> comparable ridge shrinkage
    y = np.array([r["correct"] for r in rows], int)
    return {f"C={C}": float(LogisticRegression(C=C, max_iter=3000).fit(Xs, y).coef_[0][feats.index("n_independent")])
            for C in Cs}


def beta_ind(rows, feats):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["correct"] for r in rows], int)
    return float(LogisticRegression(C=1e6, max_iter=3000).fit(X, y).coef_[0][feats.index("n_independent")])


def run():
    rows = e28.build_candidates_peer()
    res = {}

    # (B) collinearity
    res["vif_ind_sup"] = vif(rows, ["n_independent", "n_support"])
    res["vif_ind_adopt_per"] = vif(rows, ["n_independent", "n_adopt", "n_persist"])
    res["ridge_path_bind_ind_sup"] = ridge_path(rows, ["n_independent", "n_support"])
    # (C) drop-persistence ablation
    res["ablation_bind"] = {
        "ctrl_support {ind,sup}": beta_ind(rows, ["n_independent", "n_support"]),
        "decomp_with_persist {ind,adopt,per}": beta_ind(rows, ["n_independent", "n_adopt", "n_persist"]),
        "decomp_drop_persist {ind,adopt}": beta_ind(rows, ["n_independent", "n_adopt"]),
    }

    # (D) data counts + (E) pool fragmentation  (needs per-debate reconstruction)
    counts, poolsz = {}, defaultdict(list)
    corr_err = defaultdict(lambda: {"nq": 0, "all_wrong": 0, "agent_wrong": defaultdict(int), "agent_n": defaultdict(int)})
    dcr_rows = defaultdict(lambda: {"fin": 0, "hist": 0, "maj": 0, "n": 0})
    from collections import Counter
    for cfg in e26.CONFIGS:
        for b in e26.BENCHES:
            ld = e26.logdir(cfg, b)
            files = sorted(glob.glob(f"{ld}/{b}_seed*.jsonl"))
            if not files:
                continue
            nq, ncand = set(), 0
            for f in files:
                for lg in read_jsonl(f):
                    p = e3.prep_log(lg, b); F, agents = p["F"], p["agents"]; B, C = p["B"], p["C"]
                    cells = [(r, a) for r in range(F + 1) for a in agents if B[r][a] is not None]
                    pool = {B[r][a] for r, a in cells}
                    nq.add(lg["question_id"]); ncand += len(pool)
                    poolsz[b].append(len(pool))
                    # correlated error at round 0 (only agents that produced a round-0 answer)
                    ce = corr_err[b]; r0 = [a for a in agents if B[0][a] is not None]
                    if r0:
                        ce["nq"] += 1
                        wrongs = [0 if C[0][a] else 1 for a in r0]
                        if all(w == 1 for w in wrongs):
                            ce["all_wrong"] += 1
                        for a, w in zip(r0, wrongs):
                            ce["agent_wrong"][a] += w; ce["agent_n"][a] += 1
                    # dCR / majority (core benches, E_orig only -> matches tab:preserve scope)
                    if cfg == "E_orig" and b in CORE:
                        d = dcr_rows[b]; d["n"] += 1
                        gt = p.get("gt"); corrset = {B[r][a] for r in range(F + 1) for a in agents if B[r][a] is not None and C[r][a]}
                        finpool = {B[F][a] for a in agents if B[F][a] is not None}
                        d["hist"] += 1 if corrset else 0
                        d["fin"] += 1 if (corrset & finpool) else 0
                        fb = [B[F][a] for a in agents if B[F][a] is not None]
                        if fb:
                            win = Counter(fb).most_common(1)[0][0]
                            d["maj"] += 1 if any(C[F][a] for a in agents if B[F][a] == win) else 0
            counts[f"{cfg}|{b}"] = {"n_seeds": len(files), "n_questions": len(nq), "n_candidates": ncand}
    res["data_counts"] = counts
    res["pool_size_by_bench"] = {b: round(float(np.mean(v)), 2) for b, v in poolsz.items()}

    # (F) correlated error summary
    ce_out = {}
    for b, ce in corr_err.items():
        if ce["nq"] < 20:
            continue
        marg = [ce["agent_wrong"][a] / ce["agent_n"][a] for a in ce["agent_n"] if ce["agent_n"][a] > 0]
        indep_pred = float(np.prod(marg)) if marg else float("nan")
        obs = ce["all_wrong"] / ce["nq"]
        ce_out[b] = {"obs_all_wrong": round(obs, 3), "indep_pred_all_wrong": round(indep_pred, 3),
                     "ratio_obs_over_indep": round(obs / indep_pred, 2) if indep_pred > 0 else None,
                     "mean_agent_err": round(float(np.mean(marg)), 3)}
    res["correlated_error_round0"] = ce_out

    # (G) dCR vs baseline accuracy (E_orig core, per-seed pooled to match tab:preserve)
    dcr_vs_acc = {}
    for b, d in dcr_rows.items():
        if d["n"]:
            dcr_vs_acc[b] = {"maj_acc": round(d["maj"] / d["n"], 3), "CR_fin": round(d["fin"] / d["n"], 3),
                             "CR_hist": round(d["hist"] / d["n"], 3), "dCR": round((d["hist"] - d["fin"]) / d["n"], 3)}
    xs = np.array([v["maj_acc"] for v in dcr_vs_acc.values()])
    ys = np.array([v["dCR"] for v in dcr_vs_acc.values()])
    res["dcr_vs_accuracy"] = {"per_bench": dcr_vs_acc,
                              "pearson_r": round(float(np.corrcoef(xs, ys)[0, 1]), 3) if len(xs) > 2 else None}

    # ---- print ----
    print("=" * 70)
    print("(B) VIF  {ind,sup}:", {k: round(v, 2) for k, v in res["vif_ind_sup"].items()},
          " {ind,adopt,per}:", {k: round(v, 2) for k, v in res["vif_ind_adopt_per"].items()})
    print("    ridge path b_ind (standardized, {ind,sup}):", {k: round(v, 3) for k, v in res["ridge_path_bind_ind_sup"].items()})
    print("(C) ablation b_ind:", {k: round(v, 3) for k, v in res["ablation_bind"].items()})
    print("(E) pool size by bench:", res["pool_size_by_bench"])
    print("(F) correlated error @round0:")
    for b, v in ce_out.items():
        print(f"    {b:14} obs_all_wrong={v['obs_all_wrong']}  indep_pred={v['indep_pred_all_wrong']}  "
              f"ratio={v['ratio_obs_over_indep']}  mean_agent_err={v['mean_agent_err']}")
    print("(G) dCR vs maj_acc  pearson_r =", res["dcr_vs_accuracy"]["pearson_r"])
    for b, v in dcr_vs_acc.items():
        print(f"    {b:14} maj={v['maj_acc']}  dCR={v['dCR']}")
    print(f"(D) data-count cells: {len(counts)}  (see json)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[OK] {OUT}")
    return res


if __name__ == "__main__":
    run()
