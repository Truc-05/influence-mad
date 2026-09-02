"""
E3 — Two-stage behavioral failure-mode router (ZERO-GPU, early-warning).

Stage A (hazard detector): P(future behavioral extinction | H_{<=r_obs})
    positive = a correct hypothesis is alive now (r_obs) but lost behaviorally
               later (t* > r_obs); negative = alive now and survives to final.
    censored/parser-artifact deaths are excluded from labels.

Stage B (mechanism router): on behavioral-extinction events observed at t*-1,
    P(social | risk) vs P(self-error | risk).

OBSERVATION-TIME FREEZE (anti-retrospective): every feature is computed ONLY from
rounds[0 .. r_obs]. `_features()` slices the transcript to rounds[:r_obs+1] and
asserts it never touches a later round. We never use: final answers, future
adoption/changes, gold correctness of any agent, or "who holds the correct
answer". Labels use gold; features never do.

Feature blocks (for anti-conference ablation):
    f1_ structural geometry   (strictly causal)
    f2_ behavioral dynamics   (strictly causal)
    f3_ influence/history     (conference PageRank/stubbornness/M_ij; GLOBAL per
                               seed -> an OPTIMISTIC, mildly leaky upper bound for
                               the influence block, on purpose)
    f4_ text/reasoning        (strictly causal, cheap; no embeddings)

Output: results/tables/e3_router.json
"""
import json, glob, os, re, sys
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, confusion_matrix)

from src.utils.io_utils import read_jsonl
from src.utils.scoring import is_correct
from src.utils.answer_extract import robust_answer, _degenerate
from src.influence_estimation.influence_matrix import build_influence_matrix
from src.influence_estimation.pagerank_score import compute_influence_scores
from experiments.e4a_extinction_anatomy import bucket

BENCH_ORDER = ["gsm8k", "strategyqa", "commonsenseqa", "bbh"]
TAB_DIR = "results/tables"


# ----------------------------- per-log precompute -----------------------------
def prep_log(lg, bench):
    """Cache robust bucket + correctness + reasoning per round/agent."""
    gt = lg["ground_truth"]; ca = lg.get("correct_answers")
    F = len(lg["rounds"]) - 1
    agents = list(lg["rounds"][0]["answers"].keys())
    B = []; C = []; R = []; trunc = []
    for r in range(F + 1):
        bk = {}; ok = {}; rea = {}; tr = {}
        for a in agents:
            o = lg["rounds"][r]["answers"][a]
            ra = robust_answer(o["answer"], o["reasoning"], bench)
            bk[a] = bucket(ra, bench)
            ok[a] = (ra is not None) and is_correct(ra, gt, bench, ca)
            rea[a] = o["reasoning"] or ""
            tr[a] = _degenerate(o["answer"])
        B.append(bk); C.append(ok); R.append(rea); trunc.append(tr)
    ncorr = [sum(c.values()) for c in C]
    return {"agents": agents, "F": F, "B": B, "C": C, "R": R, "trunc": trunc, "ncorr": ncorr}


def extinction_of(p):
    """Return (t_star, mechanism) for a behavioral extinction, else None.
    mechanism in {'social','self'}; censored deaths return None."""
    F, C, B, ncorr, agents = p["F"], p["C"], p["B"], p["ncorr"], p["agents"]
    if not (ncorr[0] >= 1 and ncorr[F] == 0):
        return None
    s = F
    while s - 1 >= 1 and ncorr[s - 1] == 0:
        s -= 1
    t = s
    holders = [a for a in agents if C[t - 1][a]]
    bt = B[t]; bp = B[t - 1]
    if all(bt[a] is None for a in holders):
        return None                                   # censored
    social = any(bt[a] is not None and bt[a] in {bp[o] for o in agents if o != a and bp[o] is not None}
                 for a in holders)
    return t, ("social" if social else "self")


# ------------------------------- features -------------------------------------
def _jacc(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def _features(p, r_obs, infl):
    """All features from rounds[0..r_obs] ONLY. infl = per-seed influence ctx."""
    assert r_obs < p["F"] + 1
    agents, B, C, R, trunc = p["agents"], p["B"], p["C"], p["R"], p["trunc"]
    # guard: never read a future round
    B = B[:r_obs + 1]; R = R[:r_obs + 1]; trunc = trunc[:r_obs + 1]
    cur = B[r_obs]
    vals = [cur[a] for a in agents if cur[a] is not None]
    supp = Counter(vals)
    counts = sorted(supp.values(), reverse=True) or [0]
    n_ans = len(vals)
    probs = [c / n_ans for c in supp.values()] if n_ans else []
    entropy = -sum(pp * np.log(pp) for pp in probs) if probs else 0.0
    pairs_agree = sum(1 for i in range(len(agents)) for j in range(i + 1, len(agents))
                      if cur[agents[i]] is not None and cur[agents[i]] == cur[agents[j]])
    f = {
        "f1_n_distinct": len(supp),
        "f1_largest_support": counts[0],
        "f1_support_margin": counts[0] - (counts[1] if len(counts) > 1 else 0),
        "f1_minority": int(counts[-1] == 1 and len(supp) > 1),
        "f1_disagree_ratio": len(supp) / n_ans if n_ans else 1.0,
        "f1_entropy": entropy,
        "f1_pairs_agree": pairs_agree,
        "f1_n_answered": n_ans,
    }
    # F2 behavioral dynamics (rounds 1..r_obs)
    chg = [p["_changed"][r] for r in range(1, r_obs + 1)]   # list of {agent:bool}
    total_chg = sum(sum(1 for a in agents if c.get(a)) for c in chg)
    at_last = chg[-1] if chg else {}
    n_changed_last = sum(1 for a in agents if at_last.get(a))
    # changed while peers' config stayed same between r_obs-2 and r_obs-1 (self-drift cue)
    self_drift = 0
    if r_obs >= 2:
        prev_cfg = {a: B[r_obs - 2][a] for a in agents}
        mid_cfg = {a: B[r_obs - 1][a] for a in agents}
        for a in agents:
            peers_same = all(prev_cfg[o] == mid_cfg[o] for o in agents if o != a)
            if at_last.get(a) and peers_same:
                self_drift = 1
    per_agent_chgrate = []
    if r_obs >= 1:
        for a in agents:
            opp = sum(1 for r in range(1, r_obs + 1))
            kept = sum(1 for r in range(1, r_obs + 1) if not p["_changed"][r].get(a, False))
            per_agent_chgrate.append(1 - kept / opp if opp else 0.0)
    f.update({
        "f2_revision_count": total_chg,
        "f2_revision_velocity": n_changed_last / len(agents),
        "f2_n_changed_last": n_changed_last,
        "f2_self_drift_cue": self_drift,
        "f2_max_chgrate": max(per_agent_chgrate) if per_agent_chgrate else 0.0,
        "f2_mean_chgrate": float(np.mean(per_agent_chgrate)) if per_agent_chgrate else 0.0,
    })
    # F3 conference influence (GLOBAL per seed; optimistic/leaky upper bound)
    scores, stub, M, idx = infl["scores"], infl["stub"], infl["M"], infl["idx"]
    sv = [scores[a] for a in agents]
    # influence of the current lone-dissenter (gold-free: minority by support)
    dissenter_infl = 0.0
    if len(supp) > 1 and counts[-1] == 1:
        minb = min(supp, key=supp.get)
        for a in agents:
            if cur[a] == minb:
                dissenter_infl = scores[a]
    f.update({
        "f3_max_infl": max(sv), "f3_min_infl": min(sv),
        "f3_infl_concentration": max(sv) / sum(sv) if sum(sv) else 0.0,
        "f3_dissenter_infl": dissenter_infl,
        "f3_max_stub": max(stub[a] for a in agents),
        "f3_mean_stub": float(np.mean([stub[a] for a in agents])),
        "f3_max_M": float(np.max(M)) if M.size else 0.0,
    })
    # F4 text/reasoning (rounds<=r_obs)
    rea_last = R[r_obs]
    lens = [len(rea_last[a].split()) for a in agents]
    own_overlap = []
    if r_obs >= 1:
        own_overlap = [_jacc(R[r_obs][a], R[r_obs - 1][a]) for a in agents]
    peer_overlap = [_jacc(rea_last[a], rea_last[o]) for a in agents for o in agents if a != o]
    n_numeric = [len(re.findall(r"-?\d+\.?\d*", rea_last[a])) for a in agents]
    f.update({
        "f4_mean_len": float(np.mean(lens)), "f4_max_len": float(np.max(lens)),
        "f4_len_std": float(np.std(lens)),
        "f4_own_overlap_min": float(np.min(own_overlap)) if own_overlap else 1.0,
        "f4_peer_overlap_max": float(np.max(peer_overlap)) if peer_overlap else 0.0,
        "f4_trunc_any": int(any(trunc[r_obs][a] for a in agents)),
        "f4_max_numeric": float(np.max(n_numeric)),
    })
    return f


# ------------------------------ build datasets --------------------------------
def build():
    hz, mech = [], []                                 # rows: dict(feat) + meta
    for bench in BENCH_ORDER:
        for fpath in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
            logs = read_jsonl(fpath)
            if not logs:
                continue
            agents = list(logs[0]["final_answers"].keys())
            M = build_influence_matrix(logs, agents)
            infl = {"scores": compute_influence_scores(M, agents),
                    "stub": _global_stub(logs, agents), "M": M,
                    "idx": {a: i for i, a in enumerate(agents)}}
            for lg in logs:
                p = prep_log(lg, bench)
                p["_changed"] = [None] + [lg["rounds"][r].get("changed", {})
                                          for r in range(1, p["F"] + 1)]
                ext = extinction_of(p)
                is_ext = ext is not None
                is_censored_death = (p["ncorr"][0] >= 1 and p["ncorr"][p["F"]] == 0 and not is_ext)
                t_star = ext[0] if is_ext else None
                for r_obs in range(0, p["F"]):            # need a future round to warn about
                    if p["ncorr"][r_obs] < 1:
                        continue                          # no correct hypothesis to protect
                    if is_censored_death:
                        continue                          # unlabelable
                    if is_ext and r_obs >= t_star:
                        continue                          # already lost; not early warning
                    label = 1 if (is_ext and r_obs < t_star) else (0 if p["ncorr"][p["F"]] >= 1 else None)
                    if label is None:
                        continue
                    feat = _features(p, r_obs, infl)
                    hz.append({"bench": bench, "qid": lg["question_id"], "y": label, **feat})
                if is_ext:                                # E3b sample at t*-1
                    feat = _features(p, t_star - 1, infl)
                    mech.append({"bench": bench, "qid": lg["question_id"],
                                 "y": 1 if ext[1] == "social" else 0, **feat})
    return hz, mech


def _global_stub(logs, agents):
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


# ------------------------------- evaluation -----------------------------------
def _prefixes(feat_keys, blocks):
    return [k for k in feat_keys if any(k.startswith(b) for b in blocks)]


def _oof(X, y, groups, model):
    n_splits = min(5, np.unique(groups).size, int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        return None
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _mk(name):
    if name == "lr":
        return lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    return lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1)


def eval_stageA(hz):
    keys = [k for k in hz[0] if k.startswith(("f1_", "f2_", "f3_", "f4_"))]
    y = np.array([r["y"] for r in hz]); groups = np.array([r["qid"] for r in hz])
    prev = y.mean()
    blocksets = {"F1": ["f1_"], "F1+F2": ["f1_", "f2_"], "F1+F2+F3": ["f1_", "f2_", "f3_"],
                 "F1+F2+F4": ["f1_", "f2_", "f4_"], "ALL": ["f1_", "f2_", "f3_", "f4_"]}
    out = {"prevalence": float(prev), "n": len(hz), "n_pos": int(y.sum()), "pooled": {}, "lobo": {}}
    print(f"\n#### E3a HAZARD DETECTOR (n={len(hz)}, pos={int(y.sum())}, prevalence={prev:.3f}) ####")
    print(f"{'features':10s} {'model':4s} {'AUPRC':>6} {'lift':>5} {'ROC':>5} {'Brier':>6} "
          f"{'R@P.5':>6} {'P@R.5':>6}")
    for bname, blocks in blocksets.items():
        cols = _prefixes(keys, blocks)
        X = np.array([[r[k] for k in cols] for r in hz])
        for mdl in ["lr", "hgb"]:
            oof = _oof(X, y, groups, _mk(mdl))
            if oof is None:
                continue
            ap = average_precision_score(y, oof); roc = roc_auc_score(y, oof)
            br = brier_score_loss(y, oof)
            prec, rec, thr = precision_recall_curve(y, oof)
            rAtP = max([rec[i] for i in range(len(prec)) if prec[i] >= 0.5], default=0.0)
            pAtR = max([prec[i] for i in range(len(rec)) if rec[i] >= 0.5], default=0.0)
            out["pooled"][f"{bname}|{mdl}"] = {"auprc": ap, "lift": ap / prev, "roc": roc,
                                               "brier": br, "recall@p.5": rAtP, "prec@r.5": pAtR}
            print(f"{bname:10s} {mdl:4s} {ap:>6.3f} {ap/prev:>5.2f} {roc:>5.3f} {br:>6.3f} "
                  f"{rAtP:>6.3f} {pAtR:>6.3f}")
    # LOBO with ALL features, hgb
    print("  -- LOBO (train 3 benchmarks -> test held-out), ALL feats, hgb --")
    cols = _prefixes(keys, ["f1_", "f2_", "f3_", "f4_"])
    for hb in BENCH_ORDER:
        tr = [r for r in hz if r["bench"] != hb]; te = [r for r in hz if r["bench"] == hb]
        Xtr = np.array([[r[k] for k in cols] for r in tr]); ytr = np.array([r["y"] for r in tr])
        Xte = np.array([[r[k] for k in cols] for r in te]); yte = np.array([r["y"] for r in te])
        if yte.sum() < 3 or (1 - yte).sum() < 3:
            continue
        m = _mk("hgb")(); m.fit(Xtr, ytr); pr = m.predict_proba(Xte)[:, 1]
        ap = average_precision_score(yte, pr); roc = roc_auc_score(yte, pr)
        out["lobo"][hb] = {"auprc": ap, "lift": ap / yte.mean(), "roc": roc, "prev": float(yte.mean())}
        print(f"     test={hb:14s} prev={yte.mean():.3f} AUPRC={ap:.3f} (lift {ap/yte.mean():.2f}) ROC={roc:.3f}")
    return out


def _zscore_within_bench(rows, keys):
    """z-score each feature within benchmark -> removes the |A|/benchmark-level
    confound, so the classifier cannot 'predict mechanism by predicting benchmark'
    (the same control E2 used). Returns {feature: np.array}."""
    Z = {k: np.zeros(len(rows)) for k in keys}
    idxby = defaultdict(list)
    for i, r in enumerate(rows):
        idxby[r["bench"]].append(i)
    for _, ids in idxby.items():
        for k in keys:
            v = np.array([rows[i][k] for i in ids], float)
            mu, sd = v.mean(), v.std()
            for i in ids:
                Z[k][i] = (rows[i][k] - mu) / sd if sd > 1e-9 else 0.0
    return Z


def eval_stageB(mech):
    keys = [k for k in mech[0] if k.startswith(("f1_", "f2_", "f3_", "f4_"))]
    y = np.array([r["y"] for r in mech]); groups = np.array([r["qid"] for r in mech])
    Z = _zscore_within_bench(mech, keys)     # confound-controlled feature space
    out = {"n": len(mech), "n_social": int(y.sum()), "n_self": int((1 - y).sum()),
           "note": "features z-scored WITHIN benchmark (|A| confound control, matches E2)",
           "pooled": {}, "lobo_gsm8k": {}}
    print(f"\n#### E3b MECHANISM ROUTER (n={len(mech)}, social={int(y.sum())}, "
          f"self={int((1-y).sum())}) [z-scored within benchmark] ####")
    print(f"{'features':16s} {'model':4s} {'ROC':>5} {'balAcc':>7} {'macroF1':>8} {'MCC':>6} "
          f"{'Rec_soc':>8} {'Rec_self':>9}")
    blocksets = {"influence(F3)": ["f3_"], "structure(F1)": ["f1_"],
                 "struct+behav": ["f1_", "f2_"], "+influence": ["f1_", "f2_", "f3_"],
                 "+text(ALL)": ["f1_", "f2_", "f3_", "f4_"]}
    for bname, blocks in blocksets.items():
        cols = _prefixes(keys, blocks)
        X = np.column_stack([Z[k] for k in cols])
        for mdl in ["lr", "hgb"]:
            oof = _oof(X, y, groups, _mk(mdl))
            if oof is None:
                continue
            pred = (oof >= 0.5).astype(int)
            roc = roc_auc_score(y, oof); ba = balanced_accuracy_score(y, pred)
            mf1 = f1_score(y, pred, average="macro"); mcc = matthews_corrcoef(y, pred)
            cm = confusion_matrix(y, pred, labels=[1, 0])   # rows: social, self
            rec_soc = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0
            rec_self = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0
            out["pooled"][f"{bname}|{mdl}"] = {"roc": roc, "bal_acc": ba, "macro_f1": mf1,
                                               "mcc": mcc, "recall_social": rec_soc, "recall_self": rec_self}
            print(f"{bname:16s} {mdl:4s} {roc:>5.3f} {ba:>7.3f} {mf1:>8.3f} {mcc:>6.3f} "
                  f"{rec_soc:>8.3f} {rec_self:>9.3f}")
    # LOBO: train non-gsm8k -> test gsm8k (open-answer stress test)
    cols = _prefixes(keys, ["f1_", "f2_", "f4_"])   # exclude leaky-ish F3 for the clean stress test
    tr = [r for r in mech if r["bench"] != "gsm8k"]; te = [r for r in mech if r["bench"] == "gsm8k"]
    Xtr = np.array([[r[k] for k in cols] for r in tr]); ytr = np.array([r["y"] for r in tr])
    Xte = np.array([[r[k] for k in cols] for r in te]); yte = np.array([r["y"] for r in te])
    if yte.sum() >= 3 and (1 - yte).sum() >= 3:
        m = _mk("hgb")(); m.fit(Xtr, ytr); pr = m.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, pr)
        out["lobo_gsm8k"] = {"roc": roc, "n": len(te), "n_social": int(yte.sum()), "feats": "F1+F2+F4"}
        print(f"  -- LOBO train(SQA+CSQA+BBH)->test GSM8K [F1+F2+F4, hgb]: ROC={roc:.3f} (n={len(te)}) --")
    return out


def main():
    hz, mech = build()
    A = eval_stageA(hz); B = eval_stageB(mech)
    os.makedirs(TAB_DIR, exist_ok=True)
    json.dump({"stageA": A, "stageB": B}, open(f"{TAB_DIR}/e3_router.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(f"\n[OK] {TAB_DIR}/e3_router.json")


if __name__ == "__main__":
    main()
