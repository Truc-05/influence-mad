"""
E2 — Does CONFERENCE influence discriminate social from self-error extinctions?

Not "does influence correlate with stubbornness" (the conference already showed
r=0.69). The journal question is sharper:

    Given a pre-extinction event, do the conference's influence analytics
    (PageRank score, pairwise adoption M_ij, stubbornness) separate a socially
    -induced loss from an internally-generated self-error?

Unit  : one behavioral extinction event (death transition t*-1 -> t*), robustly
        scored, labelled social (holder herds onto a peer's bucket) vs self_error
        (holder drifts to a novel wrong value). Censored/parse-fail dropped.

Two anti-footgun rules:
  (1) NON-CIRCULAR features. The label is defined by event-local adoption, so we
      use ONLY global, agent-level influence properties (per benchmark+seed):
      holder PageRank influence, holder stubbornness, max/mean peer influence,
      holder's global adoption-propensity max_p M[p][holder]; plus structural
      disagreement. We never feed the event-local "did holder copy this peer".
  (2) |A|-CONFOUND control. social-fraction tracks answer-space size, and
      influence scores differ across benchmarks. We z-score every continuous
      feature WITHIN (benchmark, seed) so the classifier learns within-benchmark
      contrasts, not the cross-benchmark base rate.

Reports: (a) gsm8k clean-anchor distribution of key influence features by class;
         (b) grouped-CV AUC for social-vs-self using influence-only vs
         structural-only vs both, with a label-permutation floor and n per class.

Gate: influence-only AUC clearly > 0.5 (and > permutation floor) => influence
carries mechanism signal, keep it as a diagnostic module. Else => demote PageRank
to a baseline and let richer mechanism features (E3) lead.

Output: results/tables/e2_mechanism_influence.json
"""
import json, glob, os, sys
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

from src.utils.io_utils import read_jsonl
from src.utils.scoring import is_correct
from src.utils.answer_extract import robust_answer
from src.influence_estimation.influence_matrix import build_influence_matrix
from src.influence_estimation.pagerank_score import compute_influence_scores
from experiments.e4a_extinction_anatomy import bucket

BENCH_ORDER = ["gsm8k", "strategyqa", "commonsenseqa", "bbh"]
TAB_DIR = "results/tables"


def _rans(lg, r, a, bench):
    o = lg["rounds"][r]["answers"][a]
    return robust_answer(o["answer"], o["reasoning"], bench)


def stubbornness(logs, agents):
    kept = {a: 0 for a in agents}; opp = {a: 0 for a in agents}
    for lg in logs:
        for rd in lg["rounds"][1:]:
            ch = rd.get("changed", {})
            for a in agents:
                if a in ch:
                    opp[a] += 1
                    if not ch[a]:
                        kept[a] += 1
    return {a: (kept[a] / opp[a] if opp[a] else 0.0) for a in agents}


def extinction_event(lg, bench):
    gt = lg["ground_truth"]; ca = lg.get("correct_answers")
    F = len(lg["rounds"]) - 1
    agents = list(lg["rounds"][0]["answers"].keys())
    corr = [{a: (_rans(lg, r, a, bench) is not None
                 and is_correct(_rans(lg, r, a, bench), gt, bench, ca)) for a in agents}
            for r in range(F + 1)]
    ncorr = [sum(c.values()) for c in corr]
    if not (ncorr[0] >= 1 and ncorr[F] == 0):
        return None
    s = F
    while s - 1 >= 1 and ncorr[s - 1] == 0:
        s -= 1
    t = s
    holders = [a for a in agents if corr[t - 1][a]]
    bt = {a: bucket(_rans(lg, t, a, bench), bench) for a in agents}
    bp = {a: bucket(_rans(lg, t - 1, a, bench), bench) for a in agents}
    if all(bt[a] is None for a in holders):
        return None                                   # censored, drop
    social = any(bt[a] is not None and bt[a] in {bp[o] for o in agents if o != a and bp[o] is not None}
                 for a in holders)
    # structural disagreement at t*-1
    wrong = [a for a in agents if not corr[t - 1][a]]
    wrong_buckets = [bp[a] for a in wrong]
    largest_wrong = max([wrong_buckets.count(x) for x in set(wrong_buckets)], default=0)
    n_distinct = len({bp[a] for a in agents if bp[a] is not None})
    h = holders[0]                                    # representative holder
    return {"holder": h, "agents": agents, "k": len(holders),
            "minority": len(holders) < largest_wrong, "n_distinct": n_distinct,
            "label": 1 if social else 0, "qid": lg["question_id"]}


def main():
    rows = []
    for bench in BENCH_ORDER:
        for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
            logs = read_jsonl(f)
            agents = list(logs[0]["final_answers"].keys()) if logs else []
            if not agents:
                continue
            M = build_influence_matrix(logs, agents)
            scores = compute_influence_scores(M, agents)
            stub = stubbornness(logs, agents)
            idx = {a: i for i, a in enumerate(agents)}
            for lg in logs:
                ev = extinction_event(lg, bench)
                if ev is None:
                    continue
                h = ev["holder"]; peers = [a for a in ev["agents"] if a != h]
                feat = {
                    # influence-only (global, non-circular)
                    "holder_infl": scores[h],
                    "holder_stub": stub[h],
                    "max_peer_infl": max(scores[p] for p in peers),
                    "mean_peer_infl": float(np.mean([scores[p] for p in peers])),
                    "infl_gap": max(scores[p] for p in peers) - scores[h],
                    "max_peer_stub": max(stub[p] for p in peers),
                    "holder_adopt_propensity": max(M[idx[p]][idx[h]] for p in peers),
                    # structural-only
                    "k": ev["k"], "minority": int(ev["minority"]), "n_distinct": ev["n_distinct"],
                }
                rows.append({"bench": bench, "seed": f, "qid": ev["qid"],
                             "label": ev["label"], **feat})

    INFL = ["holder_infl", "holder_stub", "max_peer_infl", "mean_peer_infl",
            "infl_gap", "max_peer_stub", "holder_adopt_propensity"]
    STRUCT = ["k", "minority", "n_distinct"]

    y = np.array([r["label"] for r in rows])
    groups = np.array([r["qid"] for r in rows])
    print(f"\nTotal behavioral events: {len(rows)}  | social={int(y.sum())}  self={int((1-y).sum())}")
    print("per-benchmark (social/self):")
    for b in BENCH_ORDER:
        yb = [r["label"] for r in rows if r["bench"] == b]
        print(f"  {b:14s} social={sum(yb):3d}  self={len(yb)-sum(yb):3d}")

    # z-score each continuous feature WITHIN (bench, seed)
    def zscored_matrix(feats):
        X = np.zeros((len(rows), len(feats)))
        buckets = defaultdict(list)
        for i, r in enumerate(rows):
            buckets[(r["bench"], r["seed"])].append(i)
        for _, ids in buckets.items():
            for j, fname in enumerate(feats):
                vals = np.array([rows[i][fname] for i in ids], float)
                mu, sd = vals.mean(), vals.std()
                for i in ids:
                    X[i, j] = (rows[i][fname] - mu) / sd if sd > 1e-9 else 0.0
        return X

    def cv_auc(feats, seed_shuffle=None, mask=None, max_splits=5):
        Xall = zscored_matrix(feats)
        m = np.ones(len(rows), bool) if mask is None else np.asarray(mask, bool)
        X = Xall[m]; yy = y[m].copy(); gg = groups[m]
        if seed_shuffle is not None:
            rng = np.random.RandomState(seed_shuffle); yy = rng.permutation(yy)
        n_splits = min(max_splits, np.unique(gg).size, int(yy.sum()), int((1 - yy).sum()))
        if n_splits < 2:
            return None
        gkf = GroupKFold(n_splits=n_splits)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        try:
            proba = cross_val_predict(clf, X, yy, cv=gkf, groups=gg, method="predict_proba")[:, 1]
            return roc_auc_score(yy, proba)
        except Exception as e:
            return f"err:{e}"

    print("\n#### E2 — social-vs-self discrimination (grouped-CV AUC, |A|-controlled) ####")
    auc_infl = cv_auc(INFL); auc_struct = cv_auc(STRUCT); auc_both = cv_auc(INFL + STRUCT)
    perm = [cv_auc(INFL, seed_shuffle=s) for s in range(20)]
    perm = [p for p in perm if isinstance(p, float)]
    perm_mean = float(np.mean(perm)) if perm else None
    print(f"influence-only AUC : {auc_infl}")
    print(f"structural-only AUC: {auc_struct}")
    print(f"influence+struct AUC: {auc_both}")
    print(f"permutation floor (influence, mean of {len(perm)}): {perm_mean:.3f}"
          if perm_mean is not None else "permutation floor: n/a")

    # gsm8k-only (clean anchor, balanced 16/21, no |A| label artifact), 3-fold
    gmask = np.array([r["bench"] == "gsm8k" for r in rows])
    g_infl = cv_auc(INFL, mask=gmask, max_splits=3)
    g_struct = cv_auc(STRUCT, mask=gmask, max_splits=3)
    print(f"\n[gsm8k-only, n=37] influence AUC = {g_infl}  | structural AUC = {g_struct}")

    # gsm8k clean-anchor distribution by class (Mann-Whitney)
    print("\n#### gsm8k anchor — feature medians by class (social vs self) ####")
    gs = [r for r in rows if r["bench"] == "gsm8k"]
    ys = np.array([r["label"] for r in gs])
    dist = {}
    if gs and ys.sum() > 0 and (1 - ys).sum() > 0:
        print(f"{'feature':26s} {'social_med':>11} {'self_med':>10} {'MWU_p':>8}")
        for fn in INFL:
            soc = [r[fn] for r in gs if r["label"] == 1]
            slf = [r[fn] for r in gs if r["label"] == 0]
            try:
                _, p = mannwhitneyu(soc, slf, alternative="two-sided")
            except Exception:
                p = float("nan")
            dist[fn] = {"social_median": float(np.median(soc)), "self_median": float(np.median(slf)), "mwu_p": float(p)}
            print(f"{fn:26s} {np.median(soc):>11.3f} {np.median(slf):>10.3f} {p:>8.3f}")

    def _v(x):
        return x if isinstance(x, float) else None
    verdict = ("KEEP influence as diagnostic module"
               if _v(auc_infl) and auc_infl >= 0.60 and (perm_mean is None or auc_infl > perm_mean + 0.07)
               else "WEAK — demote PageRank to baseline; mechanism features must lead")
    print(f"\n=> {verdict}")

    out = {"n_events": len(rows), "n_social": int(y.sum()), "n_self": int((1 - y).sum()),
           "auc": {"influence_only": _v(auc_infl), "structural_only": _v(auc_struct),
                   "influence_plus_struct": _v(auc_both), "permutation_floor": perm_mean},
           "gsm8k_feature_dist": dist,
           "per_bench": {b: {"social": sum(r["label"] for r in rows if r["bench"] == b),
                             "self": sum(1 - r["label"] for r in rows if r["bench"] == b)}
                         for b in BENCH_ORDER}}
    os.makedirs(TAB_DIR, exist_ok=True)
    json.dump(out, open(f"{TAB_DIR}/e2_mechanism_influence.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(f"\n[OK] {TAB_DIR}/e2_mechanism_influence.json")


if __name__ == "__main__":
    main()
