"""
E28 — H3 robustness: TRUE adoption decomposition + WITHIN-QUESTION control (ZERO-GPU).

Addresses two reviewer objections to E26's headline (independent origin > inherited support):

  (1) DEFINITION.  E26 used n_inherited = n_support - n_independent, where n_support is a
      *cumulative agent-round* count. That residual absorbs the ORIGINAL agent's own
      persistence, not only social adoption -> it is wrong to call it "socially inherited".
      Fix: use the TRUE social-adoption count already in the candidate table,
          n_adopt = n_late_adopt  (an agent SWITCHES ONTO h at round>=1, i.e. did not hold it
                                    the previous round)
      and fit the interpretable, linearly-independent basis
          logit P(Y=1) = b_ind*n_independent + b_adopt*n_adopt + b_persist*n_persist.
      "Independent origin" is then compared to GENUINE social adoption, not to a residual.

  (2) QUESTION-DIFFICULTY CONFOUND.  b_ind could just proxy "easy question -> many models get it
      independently". benchmark/config fixed effects do NOT control per-question difficulty.
      Fix: (a) CONDITIONAL LOGISTIC regression stratified by question (config|bench|qid) — a
      within-question fixed-effect logit that conditions difficulty out exactly; and
      (b) a non-parametric WITHIN-QUESTION concordance: among {correct, incorrect} hypothesis
      pairs from the SAME question (optionally at MATCHED total support), how often does the
      correct one carry more independent origins?

Reuses e26.build_candidates() (=e22.candidates_for over the 5-ensemble x 8-benchmark grid, 23,239
candidates). All bootstrap CIs cluster on (config|bench|qid). Deterministic. ZERO-GPU.

  CUDA_VISIBLE_DEVICES="" python experiments/e28_h3_robust.py
"""
import os, sys, json
from collections import defaultdict
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import glob
from sklearn.linear_model import LogisticRegression
from src.utils.io_utils import read_jsonl
import experiments.e26_independence as e26
import experiments.e3_router as e3

OUT = "results/tables/e28_h3_robust.json"
B_BOOT = 1000


def build_candidates_peer():
    """Rebuild the 5-ensemble x 8-benchmark candidate table from raw logs, computing the PEER-CONDITIONED
    social-adoption count: an agent switches onto h at round r>=1, not holding it at r-1, AND some PEER held
    h at r-1 (a genuine copy from a peer, not late self-generation). Matches the formal A_i^(r)(h) in the
    paper. Everything else (n_ind, n_support, n_persist, correctness, cluster) matches e22/e26."""
    rows = []
    for cfg in e26.CONFIGS:
        for b in e26.BENCHES:
            ld = e26.logdir(cfg, b)
            for f in sorted(glob.glob(f"{ld}/{b}_seed*.jsonl")):
                for lg in read_jsonl(f):
                    p = e3.prep_log(lg, b)
                    F, agents = p["F"], p["agents"]
                    B, C = p["B"], p["C"]
                    correct = {B[r][a] for r in range(F + 1) for a in agents
                               if B[r][a] is not None and C[r][a]}
                    cells = [(r, a) for r in range(F + 1) for a in agents if B[r][a] is not None]
                    pool = {B[r][a] for r, a in cells}
                    for h in pool:
                        pres = sorted({r for r, a in cells if B[r][a] == h})
                        n_adopt_peer = sum(
                            1 for r in range(1, F + 1) for a in agents
                            if B[r][a] == h and B[r - 1][a] != h
                            and any(B[r - 1][j] == h for j in agents if j != a))
                        n_adopt_naive = sum(1 for r in range(1, F + 1) for a in agents
                                            if B[r][a] == h and B[r - 1][a] != h)
                        rows.append({
                            "bench": b, "config": cfg, "qid": lg["question_id"],
                            "cluster": f"{cfg}|{b}|{lg['question_id']}",
                            "correct": int(h in correct),
                            "n_independent": sum(1 for a in agents if B[0][a] == h),
                            "n_support": sum(1 for r, a in cells if B[r][a] == h),
                            "n_final": sum(1 for a in agents if B[F][a] == h),   # FINAL-round vote count
                            "n_rounds": F + 1,                                    # realized rounds (early-stop aware)
                            "n_persist": len(pres), "first_round": pres[0],
                            "n_adopt": n_adopt_peer, "n_adopt_naive": n_adopt_naive,
                        })
    return rows


def _fit(X, y):
    if len(set(y)) < 2:
        return None
    return LogisticRegression(C=1e6, max_iter=3000, solver="lbfgs").fit(X, y).coef_[0]


def decomposition_ci(rows, feats, B=B_BOOT, seed=0, cluster_of=None):
    """Full fit + clustered bootstrap CI for every coef, plus the ind-minus-adopt contrast.
    cluster_of(r) -> cluster key (default r['cluster'] = config|bench|qid)."""
    if cluster_of is None:
        cluster_of = lambda r: r["cluster"]
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["correct"] for r in rows], int)
    full = _fit(X, y)
    cl = defaultdict(list)
    for i, r in enumerate(rows):
        cl[cluster_of(r)].append(i)
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
    # contrast: independent - adopt (per-vote log-odds gap), bootstrap CI on the difference
    if "n_independent" in feats and "n_adopt" in feats:
        ii, ia = feats.index("n_independent"), feats.index("n_adopt")
        diff = boot[:, ii] - boot[:, ia]
        out["contrast_ind_minus_adopt"] = {
            "beta": float(full[ii] - full[ia]), "lo": float(np.percentile(diff, 2.5)),
            "hi": float(np.percentile(diff, 97.5)), "p_le0": float(np.mean(diff <= 0))}
    return out


def conditional_logit(rows, feats):
    """Within-question fixed-effect (conditional) logistic; difficulty conditioned out exactly."""
    from statsmodels.discrete.conditional_models import ConditionalLogit
    y = np.array([r["correct"] for r in rows], int)
    X = np.array([[r[f] for f in feats] for r in rows], float)
    groups = np.array([r["cluster"] for r in rows])
    res = ConditionalLogit(y, X, groups=groups).fit(disp=0, method="bfgs")
    ci = res.conf_int()
    out = {}
    for k, f in enumerate(feats):
        out[f] = {"beta": float(res.params[k]), "lo": float(ci[k][0]),
                  "hi": float(ci[k][1]), "p": float(res.pvalues[k])}
    # informative strata = questions with both a correct and an incorrect candidate
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["cluster"]][r["correct"]] += 1
    out["_n_informative_questions"] = sum(1 for a, b in by.values() if a > 0 and b > 0)
    return out


def within_q_concordance(rows, match_field=None, B=2000, seed=0):
    """Among {correct, incorrect} hypotheses of the SAME question, how often does the correct one
    have MORE independent origins? 0.5 = null. match_field in {None,'n_support','n_final'} -> only pairs
    at equal value of that field (controls magnitude at cumulative or final-round resolution)."""
    by = defaultdict(list)
    for r in rows:
        by[r["cluster"]].append(r)
    clusters = []
    for rs in by.values():
        cor = [x for x in rs if x["correct"]]
        inc = [x for x in rs if not x["correct"]]
        if not cor or not inc:
            continue
        pairs = []
        for c in cor:
            for w in inc:
                if match_field is not None and c[match_field] != w[match_field]:
                    continue
                d = c["n_independent"] - w["n_independent"]
                pairs.append(1.0 if d > 0 else (0.5 if d == 0 else 0.0))
        if pairs:
            clusters.append(pairs)
    allp = [p for cl in clusters for p in cl]
    if not allp:
        return None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        idx = rng.integers(0, len(clusters), len(clusters))
        pp = [p for j in idx for p in clusters[j]]
        boot.append(np.mean(pp))
    return {"concordance": float(np.mean(allp)), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "n_clusters": len(clusters), "n_pairs": len(allp)}


def run():
    rows = build_candidates_peer()               # PEER-CONDITIONED social adoption (matches paper A_i^(r))
    y = np.array([r["correct"] for r in rows], int)
    na = np.array([r["n_adopt"] for r in rows], float)
    nan = np.array([r["n_adopt_naive"] for r in rows], float)
    ni = np.array([r["n_independent"] for r in rows], float)
    print(f"candidates={len(rows)}  base P(correct)={y.mean():.3f}")
    print(f"corr(n_ind, n_adopt)={np.corrcoef(ni, na)[0,1]:.3f}   mean n_adopt(peer)={na.mean():.3f}  "
          f"mean n_adopt(naive)={nan.mean():.3f}  mean n_ind={ni.mean():.2f}")
    print(f"peer/naive adoption agreement: {float((na==nan).mean()):.3f} of candidates identical\n")

    def fmt(d):
        return f"{d['beta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]{'*' if d['lo']>0 else ' '}"

    # ---- (1) TRUE decomposition: independent vs GENUINE adoption vs persistence ----
    dec = decomposition_ci(rows, ["n_independent", "n_adopt", "n_persist"], seed=1)
    print("=" * 72 + "\n(1) TRUE decomposition  logit = b_ind*n_ind + b_adopt*n_adopt + b_persist*n_persist\n" + "=" * 72)
    print(f"  b_independent           {fmt(dec['n_independent'])}")
    print(f"  b_adopt (true social)   {fmt(dec['n_adopt'])}")
    print(f"  b_persist               {fmt(dec['n_persist'])}")
    print(f"  contrast ind - adopt    {fmt(dec['contrast_ind_minus_adopt'])}")
    ratio = dec["n_independent"]["beta"] / dec["n_adopt"]["beta"] if dec["n_adopt"]["beta"] else float("nan")
    print(f"  log-odds coef ratio ind:adopt = {ratio:.2f}x  (report as 'coefficient N x larger', NOT 'N x value')")

    # sanity: also the two-term ind vs adopt (no persist), to compare with E27
    dec2 = decomposition_ci(rows, ["n_independent", "n_adopt"], seed=2)
    print(f"  [2-term] b_ind {fmt(dec2['n_independent'])}  b_adopt {fmt(dec2['n_adopt'])}")

    # ---- (2a) within-question conditional logit (difficulty conditioned out) ----
    print("\n" + "=" * 72 + "\n(2a) WITHIN-QUESTION conditional logit (question fixed effects)\n" + "=" * 72)
    cl_sup = conditional_logit(rows, ["n_independent", "n_support"])
    cl_adopt = conditional_logit(rows, ["n_independent", "n_adopt", "n_persist"])
    print(f"  informative questions (mixed correct/incorrect) = {cl_sup['_n_informative_questions']}")
    print(f"  b_ind | ctrl n_support (within-Q)     {fmt(cl_sup['n_independent'])}  p={cl_sup['n_independent']['p']:.2e}")
    print(f"  b_ind | ctrl adopt+persist (within-Q) {fmt(cl_adopt['n_independent'])}  p={cl_adopt['n_independent']['p']:.2e}")
    print(f"  b_adopt (within-Q)                    {fmt(cl_adopt['n_adopt'])}")

    # ---- (2b) non-parametric within-question concordance ----
    print("\n" + "=" * 72 + "\n(2b) WITHIN-QUESTION concordance: P(correct has MORE independent origins)\n" + "=" * 72)
    conc = within_q_concordance(rows, match_field=None)
    conc_m = within_q_concordance(rows, match_field="n_support")
    conc_f = within_q_concordance(rows, match_field="n_final")
    print(f"  all pairs             conc={conc['concordance']:.3f} [{conc['lo']:.3f},{conc['hi']:.3f}]  "
          f"({conc['n_clusters']} q, {conc['n_pairs']} pairs)")
    print(f"  matched cumul support conc={conc_m['concordance']:.3f} [{conc_m['lo']:.3f},{conc_m['hi']:.3f}]  "
          f"({conc_m['n_clusters']} q, {conc_m['n_pairs']} pairs)")
    print(f"  matched FINAL support conc={conc_f['concordance']:.3f} [{conc_f['lo']:.3f},{conc_f['hi']:.3f}]  "
          f"({conc_f['n_clusters']} q, {conc_f['n_pairs']} pairs)")

    # ---- (3) robustness: consensus definition, dependence clustering, exposure time ----
    print("\n" + "=" * 72 + "\n(3) ROBUSTNESS of b_ind\n" + "=" * 72)
    r_final = decomposition_ci(rows, ["n_independent", "n_final"], seed=5)          # match FINAL-round vote count
    r_bq = decomposition_ci(rows, ["n_independent", "n_support"], seed=6,
                            cluster_of=lambda r: f"{r['bench']}|{r['qid']}")        # conservative: qids shared across configs
    r_rounds = decomposition_ci(rows, ["n_independent", "n_support", "n_rounds"], seed=7)  # control realized rounds (early stop)
    print(f"  b_ind | ctrl FINAL-round support  {fmt(r_final['n_independent'])}   (deployed consensus definition)")
    print(f"  b_ind | ctrl n_support, cluster=(bench,qid)  {fmt(r_bq['n_independent'])}   (shared-question dependence)")
    print(f"  b_ind | ctrl n_support + n_rounds  {fmt(r_rounds['n_independent'])}   (exposure-time confound)")

    result = {
        "n_candidates": len(rows), "base_rate": float(y.mean()),
        "corr_ind_adopt": float(np.corrcoef(ni, na)[0, 1]),
        "true_decomposition": dec, "two_term_ind_adopt": dec2,
        "within_question_condlogit_ctrl_support": cl_sup,
        "within_question_condlogit_ctrl_adopt_persist": cl_adopt,
        "within_question_concordance_all": conc,
        "within_question_concordance_matched_support": conc_m,
        "within_question_concordance_matched_final": conc_f,
        "robust_ctrl_final_support": r_final,
        "robust_cluster_bench_qid": r_bq,
        "robust_ctrl_rounds": r_rounds,
        "log_odds_ratio_ind_over_adopt": ratio,
        "_note": ("H3 robustness. (1) TRUE decomposition replaces the flawed n_inherited residual with "
                  "n_adopt=n_late_adopt (genuine social-adoption events): b_ind vs b_adopt is the clean "
                  "independent-vs-inherited contrast. (2) within-question conditional logit + concordance "
                  "condition question difficulty out exactly; b_ind / concordance staying positive locks "
                  "the difficulty-confound objection. Report log-odds COEFFICIENT ratios, not 'value'."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")
    return result


if __name__ == "__main__":
    run()
