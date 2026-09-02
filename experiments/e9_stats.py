"""
E9 — Statistical robustness of the hazard-guided controller (ZERO-GPU).
"Is 66.1% vs 65.5% signal or noise? Does the detector really beat random?"

Resampling unit = QUESTION, not debate. The E6 precompute is cached per (bench,qid),
so all seed-events of a question share one {control,combined} outcome; the 516 events
cluster into 184 independent questions. We therefore CLUSTER-bootstrap: resample the
184 questions with replacement and recompute every statistic, which correctly inflates
the CIs for the within-question correlation. The deployed OOF hazard is treated as a
fixed scorer (we bound the evaluation variance of the shipped detector, not its
training variance). Ties in the top-k ranking are broken at random per replicate.

Reported (95% percentile CI over B replicates), pooled and per-benchmark:
  hazard - random   @ 20/30/40% budget      (detector value; the core claim)
  AUC[20-40]         hazard - random          (threshold-free detector value)
  hazard - always    @ 85% budget            (the +0.6pp peak; expected to span 0)
  rescue rate (help@would-die), harm rate (harm@would-live)

Output: results/tables/e9_stats.json , results/figures/e9_stats.png
"""
import os, sys, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.e6_offline as off

OUT = "results/tables/e9_stats.json"
FIG = "results/figures/e9_stats.png"
B = 4000
BUDGETS = [0.20, 0.30, 0.40]
PEAK_BUDGET = 0.85
INK = "#2B2B2B"
BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#8C8C8C"


def topk_fire(score, k, rng):
    n = len(score)
    order = np.lexsort((rng.random(n), -score))
    fire = np.zeros(n, bool); fire[order[:k]] = True
    return fire


def stats_on(idx, h, yc, yk, rng):
    """All statistics on an index set (idx may contain duplicates from clustering)."""
    h, yc, yk = h[idx], yc[idx], yk[idx]
    n = len(idx)
    never, always = float(yc.mean()), float(yk.mean())
    out = {"never": never, "always": always, "always_minus_never": always - never}
    for b in BUDGETS + [PEAK_BUDGET]:
        k = int(round(b * n))
        hz = float(np.mean(np.where(topk_fire(h, k, rng), yk, yc)))
        rd = float(np.mean(np.where(topk_fire(np.zeros(n), k, rng), yk, yc)))
        out[f"hz@{b}"] = hz; out[f"rd@{b}"] = rd
        out[f"hz_minus_rd@{b}"] = hz - rd
    out["hz_minus_always@peak"] = out[f"hz@{PEAK_BUDGET}"] - always
    # AUC over [0.20,0.40]
    grid = np.round(np.arange(0.20, 0.401, 0.04), 3)
    hzc = [float(np.mean(np.where(topk_fire(h, int(round(b * n)), rng), yk, yc))) for b in grid]
    rdc = [float(np.mean(np.where(topk_fire(np.zeros(n), int(round(b * n)), rng), yk, yc))) for b in grid]
    out["auc_hz"] = float(np.mean(hzc)); out["auc_rd"] = float(np.mean(rdc))
    out["auc_hz_minus_rd"] = out["auc_hz"] - out["auc_rd"]
    # rescue / harm on the natural population
    wl = yc.astype(bool)                        # would-live (control correct)
    out["harm@would_live"] = float(np.mean(1 - yk[wl])) if wl.any() else 0.0
    out["rescue@would_die"] = float(np.mean(yk[~wl])) if (~wl).any() else 0.0
    return out


def bootstrap(events, rng, B=B):
    h = np.array([e["hazard"] for e in events])
    yc, yk = off.outcome_vectors(events)
    clusters = defaultdict(list)
    for i, e in enumerate(events):
        clusters[(e["bench"], e["qid"])].append(i)
    ckeys = list(clusters.values())
    # stable point estimate: average the full-data statistic over many tie-break draws
    full = np.arange(len(events))
    acc = None
    for _ in range(200):
        s = stats_on(full, h, yc, yk, rng)
        if acc is None:
            acc = {k: [] for k in s if isinstance(s[k], float)}
        for k in acc:
            acc[k].append(s[k])
    point = {k: float(np.mean(v)) for k, v in acc.items()}
    keys = list(point.keys())
    draws = {k: np.empty(B) for k in keys}
    for b in range(B):
        pick = rng.integers(0, len(ckeys), len(ckeys))
        idx = np.concatenate([ckeys[j] for j in pick])
        s = stats_on(idx, h, yc, yk, rng)
        for k in keys:
            draws[k][b] = s[k]
    ci = {k: {"point": point[k],
              "lo": float(np.percentile(draws[k], 2.5)),
              "hi": float(np.percentile(draws[k], 97.5)),
              "p_le_0": float(np.mean(draws[k] <= 0))} for k in keys}
    return ci


def main():
    events = off.load_events()
    h, _ = off.oof_hazard(events, ("f1_", "f2_", "f3_", "f4_"))
    for e, v in zip(events, h):
        e["hazard"] = float(v)
    rng = np.random.default_rng(0)

    scopes = {"pooled": events,
              "gsm8k": [e for e in events if e["bench"] == "gsm8k"],
              "bbh":   [e for e in events if e["bench"] == "bbh"]}
    res = {"B": B, "n_questions": {s: len({(e["bench"], e["qid"]) for e in evs})
                                   for s, evs in scopes.items()}}
    for s, evs in scopes.items():
        res[s] = bootstrap(evs, np.random.default_rng(0), B)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    def show(label, d):
        star = "  *" if (d["lo"] > 0 or d["hi"] < 0) else "   "
        print(f"  {label:26s} {d['point']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]{star}")

    for s in ("pooled", "gsm8k", "bbh"):
        d = res[s]
        print(f"\n=== {s}  (n_questions={res['n_questions'][s]}) ===")
        for b in BUDGETS:
            show(f"hazard−random @{int(b*100)}%", d[f"hz_minus_rd@{b}"])
        show("AUC[20-40] hazard−random", d["auc_hz_minus_rd"])
        show("hazard−always @85%", d["hz_minus_always@peak"])
        print(f"  rescue@would-die  {d['rescue@would_die']['point']:.3f} "
              f"[{d['rescue@would_die']['lo']:.3f}, {d['rescue@would_die']['hi']:.3f}]   "
              f"harm@would-live {d['harm@would_live']['point']:.3f} "
              f"[{d['harm@would_live']['lo']:.3f}, {d['harm@would_live']['hi']:.3f}]")
    print("  (* = 95% CI excludes 0)")

    # ------------------------------- figure: forest -------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#C7CFD8", "legend.frameon": False})
    rows = [("hazard−random @20%", "hz_minus_rd@0.2"),
            ("hazard−random @30%", "hz_minus_rd@0.3"),
            ("hazard−random @40%", "hz_minus_rd@0.4"),
            ("AUC[20-40] hazard−random", "auc_hz_minus_rd"),
            ("hazard−always @85%", "hz_minus_always@peak")]
    scope_off = {"pooled": 0.0, "gsm8k": 0.26, "bbh": -0.26}
    scope_col = {"pooled": BLUE, "gsm8k": GREEN, "bbh": RED}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                   gridspec_kw={"width_ratios": [3, 2]})

    # left: forest of Δ-accuracy comparisons
    axL.axvline(0, color=INK, lw=1.1, zorder=1)
    yb = np.arange(len(rows))[::-1]
    for s in ("pooled", "gsm8k", "bbh"):
        d = res[s]; ys = yb + scope_off[s]
        for y, (_, k) in zip(ys, rows):
            lo, hi = d[k]["lo"], d[k]["hi"]
            axL.plot([lo, hi], [y, y], color=scope_col[s], lw=1.9, zorder=2)
            for xx in (lo, hi):
                axL.plot([xx, xx], [y - 0.08, y + 0.08], color=scope_col[s], lw=1.3, zorder=2)
        axL.scatter([d[k]["point"] for _, k in rows], ys, s=34, color=scope_col[s],
                    edgecolor="white", linewidth=0.6, zorder=4, label=s)
    axL.set_yticks(yb); axL.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    axL.set_xlabel("Δ accuracy  (95% cluster-bootstrap CI over questions)")
    axL.set_title("A  Detector edge over random is borderline (gsm8k-driven);\n"
                  "hazard−always margin is within noise", fontsize=9.8)
    axL.legend(loc="lower right", fontsize=9)
    axL.grid(True, axis="x", alpha=0.28)

    # right: rescue vs harm rates (the robustly significant asymmetry)
    axR.axvline(0, color=INK, lw=0.8)
    yp = {"pooled": 2, "gsm8k": 1, "bbh": 0}
    for s in ("pooled", "gsm8k", "bbh"):
        d = res[s]
        for key, col, dy, lab in [("rescue@would_die", GREEN, 0.16, "rescue (would-die)"),
                                  ("harm@would_live", RED, -0.16, "harm (would-live)")]:
            y = yp[s] + dy
            lo, hi, pt = d[key]["lo"], d[key]["hi"], d[key]["point"]
            axR.plot([lo, hi], [y, y], color=col, lw=2.0, zorder=2)
            for xx in (lo, hi):
                axR.plot([xx, xx], [y - 0.06, y + 0.06], color=col, lw=1.3)
            axR.scatter([pt], [y], s=40, color=col, edgecolor="white", linewidth=0.6,
                        zorder=4, label=lab if s == "pooled" else None)
    axR.set_yticks(list(yp.values())); axR.set_yticklabels(list(yp.keys()), fontsize=9.5)
    axR.set_xlabel("Rate under always-on safeguard  (95% CI)")
    axR.set_title("B  Intervention is significantly asymmetric:\n"
                  "rescue ≫ collateral harm", fontsize=9.8)
    axR.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axR.set_xlim(0, 0.8); axR.legend(loc="lower right", fontsize=8.5)
    axR.grid(True, axis="x", alpha=0.28)

    fig.suptitle("E9 — cluster-bootstrap (184 questions): the principle is solid, "
                 "the detector's edge is promising but underpowered", fontsize=10.8, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"\n[OK] {OUT}\n[OK] {FIG}")


if __name__ == "__main__":
    main()
