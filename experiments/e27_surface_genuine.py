"""
E27 — Surface vs genuine inherited support (ZERO-GPU MECHANISM EXTENSION).

SECONDARY / exploratory analysis that EXTENDS the frozen E26 principle; it does NOT modify E26/H3.
If it survives strongly the story deepens to  INDEPENDENT > GENUINE-inherited > SURFACE-inherited;
if it dies, E26/H3 is unaffected and this goes to robustness/appendix. (Methodological hygiene:
E26 was frozen BEFORE this was run.)

Question: inherited (late-adopted) support splits into
    surface adoption  = agent copies a peer's ANSWER but its REASONING does not converge
    genuine adoption  = the adopting agent's reasoning genuinely converges with the peer's
Test:   P(correct | genuine) > P(correct | surface)     [adoption-event level]
and the per-vote ordering  b_independent >= b_genuine >= b_surface   [candidate-level decomposition].

Convergence machinery REUSED from E24 (zero-GPU, lexical): _jacc on reasoning (p["R"]), residualized
against the debate's own baseline pairwise similarity (topic control; grounds E3b-v3's ~73% surface).
Per adoption event (a switches TO h at round r>=1): compare a's reasoning_r to the peers holding h at
round r, residualize by baseline; genuine iff residual > 0 (reasoning overlap beats the debate's own
average), else surface. Same candidate grid as E26 (5 ensembles x 8 benchmarks, MATH removed); clustered
bootstrap on config|bench|qid.

  CUDA_VISIBLE_DEVICES="" python experiments/e27_surface_genuine.py
"""
import os, sys, glob, json
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from src.utils.io_utils import read_jsonl
import experiments.e3_router as e3

OUT = "results/tables/e27_surface_genuine.json"
FIG = "results/figures/e27_surface_genuine.png"
CORE = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa"]
NEW = ["aqua", "mmlu", "arc"]
BENCHES = CORE + NEW
CONFIGS = ["E_orig", "E_7b", "E_mixed", "E_strong", "E_n5"]
B_POOL, B_CELL = 800, 500


def logdir(cfg, b):
    if cfg == "E_orig":
        return "data/debate_logs" if (b in CORE or b == "aqua") else "data/debate_logs/gen/E_orig3b"
    return f"data/debate_logs/gen/{cfg}"


def _jacc(a, b):
    sa, sb = set((a or "")[:500].lower().split()), set((b or "")[:500].lower().split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def build():
    """Returns (candidate rows, adoption-event rows, per-config reasoning-availability diag)."""
    cand, events = [], []
    reas_avail = defaultdict(list)
    for cfg in CONFIGS:
        for bench in BENCHES:
            ld = logdir(cfg, bench)
            for f in sorted(glob.glob(f"{ld}/{bench}_seed*.jsonl")):
                for lg in read_jsonl(f):
                    p = e3.prep_log(lg, bench)
                    F, agents = p["F"], p["agents"]
                    cells = [(r, a) for r in range(F + 1) for a in agents if p["B"][r][a] is not None]
                    if not cells:
                        continue
                    qid = lg["question_id"]
                    clu = f"{cfg}|{bench}|{qid}"
                    correct = {b for r, a in cells if p["C"][r][a] for b in [p["B"][r][a]]}
                    pool = {p["B"][r][a] for r, a in cells}
                    freq = Counter(p["B"][r][a] for r, a in cells)
                    reas = [p["R"][r][a] for r, a in cells]
                    reas_avail[cfg].append(float(np.mean([len((x or "").split()) for x in reas])))
                    base = (float(np.mean([_jacc(x, y) for x, y in combinations(reas, 2)]))
                            if len(reas) > 1 else 0.0)
                    for h in pool:
                        pr = sorted({r for r, a in cells if p["B"][r][a] == h})
                        n_ind = sum(1 for a in agents if p["B"][0][a] == h)
                        n_gen = n_surf = 0
                        for r in range(1, F + 1):
                            for a in agents:
                                if p["B"][r][a] == h and p["B"][r - 1][a] != h:   # a ADOPTS h at round r
                                    peers = [p["R"][r][b] for b in agents if b != a and p["B"][r][b] == h]
                                    if not peers:   # fall back to any other supporter of h across rounds
                                        peers = [p["R"][rr][bb] for rr, bb in cells
                                                 if p["B"][rr][bb] == h and not (rr == r and bb == a)]
                                    sim = float(np.mean([_jacc(p["R"][r][a], q) for q in peers])) if peers else 0.0
                                    genuine = (sim - base) > 0.0
                                    n_gen += int(genuine)
                                    n_surf += int(not genuine)
                                    events.append({"cfg": cfg, "bench": bench, "cluster": clu,
                                                   "correct": int(h in correct), "genuine": int(genuine)})
                        cand.append({"cfg": cfg, "bench": bench, "cluster": clu,
                                     "correct": int(h in correct), "n_independent": n_ind,
                                     "n_genuine_adopt": n_gen, "n_surface_adopt": n_surf,
                                     "n_late": n_gen + n_surf, "n_support": freq[h]})
    return cand, events, {k: float(np.mean(v)) for k, v in reas_avail.items()}


def _clusters(rows):
    cl = defaultdict(list)
    for i, r in enumerate(rows):
        cl[r["cluster"]].append(i)
    return list(cl.values())


def event_contrast(rows, B, seed=0):
    """P(correct|genuine) vs P(correct|surface) + clustered-boot CI on the difference."""
    g = np.array([r["correct"] for r in rows if r["genuine"]], float)
    s = np.array([r["correct"] for r in rows if not r["genuine"]], float)
    if len(g) < 5 or len(s) < 5:
        return None
    y = np.array([r["correct"] for r in rows]); gen = np.array([r["genuine"] for r in rows])
    keys = _clusters(rows); rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        pick = np.concatenate([keys[j] for j in rng.integers(0, len(keys), len(keys))])
        yy, gg = y[pick], gen[pick]
        if gg.sum() and (~gg.astype(bool)).sum():
            d.append(yy[gg == 1].mean() - yy[gg == 0].mean())
    d = np.array(d)
    return {"p_genuine": float(g.mean()), "n_genuine": len(g),
            "p_surface": float(s.mean()), "n_surface": len(s),
            "delta": float(g.mean() - s.mean()),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_le0": float(np.mean(d <= 0))}


def decomposition(rows, feats, B, seed=1):
    """Logistic coefs + clustered-boot CI for each feature (per-vote log-odds)."""
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["correct"] for r in rows], int)
    if len(set(y)) < 2:
        return None
    full = LogisticRegression(C=1e6, max_iter=2000).fit(X, y).coef_[0]
    keys = _clusters(rows); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        pick = np.concatenate([keys[j] for j in rng.integers(0, len(keys), len(keys))])
        yy = y[pick]
        if len(set(yy)) < 2:
            continue
        bs.append(LogisticRegression(C=1e6, max_iter=2000).fit(X[pick], yy).coef_[0])
    bs = np.array(bs)
    out = {}
    for k, f in enumerate(feats):
        out[f] = {"beta": float(full[k]), "lo": float(np.percentile(bs[:, k], 2.5)),
                  "hi": float(np.percentile(bs[:, k], 97.5))}
    # genuine-minus-surface contrast on the same bootstrap draws
    gi, si = feats.index("n_genuine_adopt"), feats.index("n_surface_adopt")
    diff = bs[:, gi] - bs[:, si]
    out["_genuine_minus_surface"] = {"beta": float(full[gi] - full[si]),
                                     "lo": float(np.percentile(diff, 2.5)),
                                     "hi": float(np.percentile(diff, 97.5)),
                                     "p_le0": float(np.mean(diff <= 0))}
    return out


def make_figure(dec, path):
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    labels = ["independent\n(round-0 origin)", "genuine\ninherited", "surface\ninherited"]
    keys = ["n_independent", "n_genuine_adopt", "n_surface_adopt"]
    betas = [dec[k]["beta"] for k in keys]
    los = [dec[k]["beta"] - dec[k]["lo"] for k in keys]
    his = [dec[k]["hi"] - dec[k]["beta"] for k in keys]
    colors = ["#2c7d59", "#5aa9d6", "#c98a3a"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(3)
    ax.bar(x, betas, yerr=[los, his], capsize=5, color=colors, width=.62, edgecolor="white")
    ax.axhline(0, color="#888", lw=.8)
    for xi, b in zip(x, betas):
        ax.text(xi, b + (0.04 if b >= 0 else -0.08), f"{b:+.2f}", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("per-vote effect on P(correct)  (log-odds)")
    gm = dec["_genuine_minus_surface"]
    sub = ("independent > genuine-inherited > surface-inherited"
           if gm["lo"] > 0 else
           "independent origin dominates; genuine ≈ surface once controlled (n.s.)")
    ax.set_title("Origin, not reasoning-convergence, is what carries the signal\n" + sub,
                 fontsize=12, loc="left")
    fig.text(0.012, 0.005, "candidate-level logistic; error bars = clustered-bootstrap 95% CI (config|bench|qid).",
             fontsize=7.6, color="#666")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run():
    cand, events, diag = build()
    print(f"candidates={len(cand)}  adoption-events={len(events)}")
    print("mean reasoning words/cell by config (sanity — must be >0 or convergence is degenerate):")
    for c, v in diag.items():
        print(f"   {c:9} {v:.1f}")
    ng = sum(e["genuine"] for e in events); tot = len(events)
    print(f"\nadoption split: genuine {ng} ({ng/tot:.1%}) / surface {tot-ng} ({(tot-ng)/tot:.1%})  "
          f"(E3b-v3 reported ~73% surface)")

    print("\n" + "=" * 72 + "\nA. EVENT-LEVEL  P(correct | genuine) vs P(correct | surface)\n" + "=" * 72)
    ec = event_contrast(events, B_POOL, seed=0)
    print(f"  POOLED  genuine {ec['p_genuine']:.3f} (n{ec['n_genuine']})  surface {ec['p_surface']:.3f} "
          f"(n{ec['n_surface']})  Δ={ec['delta']:+.3f} [{ec['lo']:+.3f},{ec['hi']:+.3f}]"
          f"{'*' if ec['lo'] > 0 else ' '} p(Δ<=0)={ec['p_le0']:.3f}")
    per_bench_ev, per_cfg_ev = {}, {}
    for b in BENCHES:
        r = event_contrast([e for e in events if e["bench"] == b], B_CELL, seed=5)
        per_bench_ev[b] = r
        if r:
            print(f"    {b:14} gen {r['p_genuine']:.3f} surf {r['p_surface']:.3f}  "
                  f"Δ={r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]{'*' if r['lo'] > 0 else ' '}")
    for c in CONFIGS:
        r = event_contrast([e for e in events if e["cfg"] == c], B_CELL, seed=6)
        per_cfg_ev[c] = r
        if r:
            print(f"    {c:14} gen {r['p_genuine']:.3f} surf {r['p_surface']:.3f}  "
                  f"Δ={r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]{'*' if r['lo'] > 0 else ' '}")

    print("\n" + "=" * 72 + "\nB. DECOMPOSITION  b_ind / b_genuine / b_surface (per-vote log-odds)\n" + "=" * 72)
    dec = decomposition(cand, ["n_independent", "n_genuine_adopt", "n_surface_adopt"], B_POOL, seed=1)
    for k in ["n_independent", "n_genuine_adopt", "n_surface_adopt"]:
        d = dec[k]
        print(f"  {k:18} {d['beta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]{'*' if d['lo'] > 0 else ' '}")
    gm = dec["_genuine_minus_surface"]
    print(f"  genuine − surface   {gm['beta']:+.3f} [{gm['lo']:+.3f},{gm['hi']:+.3f}]"
          f"{'*' if gm['lo'] > 0 else ' '} p(<=0)={gm['p_le0']:.3f}   <- the KEY test")
    order = " > ".join(k.replace("n_", "").replace("_adopt", "") for k in
                       sorted(["n_independent", "n_genuine_adopt", "n_surface_adopt"],
                              key=lambda k: -dec[k]["beta"]))
    print(f"  observed ordering:  {order}")

    make_figure(dec, FIG)
    print(f"\n[fig] {FIG}")

    verdict = ("GENUINE>SURFACE holds (event Δ CI-lo>0 AND decomp genuine−surface CI-lo>0)"
               if (ec["lo"] > 0 and gm["lo"] > 0) else
               "NOT clean — report in appendix/robustness, do NOT promote; E26/H3 unaffected")
    result = {"n_candidates": len(cand), "n_events": len(events),
              "surface_fraction": (tot - ng) / tot, "reasoning_words_by_config": diag,
              "event_contrast": {"pooled": ec, "per_bench": per_bench_ev, "per_config": per_cfg_ev},
              "decomposition": dec, "figure": FIG, "verdict": verdict,
              "_note": ("SECONDARY mechanism extension; E26/H3 frozen independently. Convergence = "
                        "lexical reasoning-jaccard residualized vs debate baseline (E24 machinery); "
                        "genuine iff per-event residual>0. Promote to a contribution ONLY if both the "
                        "event-level Δ and the decomposition genuine−surface contrast are CI-lo>0.")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"[OK] {OUT}\nVERDICT: {verdict}")
    return result


if __name__ == "__main__":
    run()
