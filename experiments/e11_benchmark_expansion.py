"""
E11 — Benchmark-expansion harness for the hazard-guided controller.

Motivation: E8 showed the controller works on GSM8K but NOT on BBH. Before spending GPU
on a full 6-benchmark counterfactual precompute, we need to know where extinction is even
*predictable* --- the necessary condition for a hazard-guided controller to help.

LAYER 1 (this script, ZERO-GPU, runnable now): within-benchmark Stage-A hazard
detectability. For every benchmark with debate logs we build the g=0 population and
measure the OOF hazard's AUPRC-lift for ranking true extinctions --- NO counterfactual
outcomes required. This is the detectability landscape.

LAYER 2 (GPU, after per-seed E6 lands, NOT run here): for each benchmark, precompute
control/combined counterfactuals (reuse experiments/e6_controller.py --bench <b>
--config config/config_e6_gpu1.yaml), then run the E6/E7/E9 offline verdict per
benchmark and assemble the cross-benchmark controller table (hazard-vs-random AUC[20-40]
with cluster-bootstrap CI, hazard-vs-always, rescue/harm). Detectability (Layer 1) is
necessary but NOT sufficient --- BBH has AUPRC lift yet the controller fails --- so
Layer 2 is what actually decides generalization.

Chosen 6-benchmark set (per project decision): GSM8K, BBH, StrategyQA, CommonsenseQA,
AQuA, MATH; MATH is a stress test (low extinction yield), not required to show signal.

Usage:  python experiments/e11_benchmark_expansion.py
Output: results/tables/e11_detectability.json , results/figures/e11_detectability.png
"""
import os, sys, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.e6_controller as e6
import experiments.e6_offline as off

OUT = "results/tables/e11_detectability.json"
FIG = "results/figures/e11_detectability.png"
BENCHES = ["gsm8k", "bbh", "strategyqa", "commonsenseqa", "truthfulqa", "aqua", "math"]
GROUPS = {"ALL": ("f1_", "f2_", "f3_", "f4_"), "F1": ("f1_",), "F3": ("f3_",), "F4": ("f4_",)}
MIN_POS = 8            # need enough extinction events for a meaningful within-bench OOF
INK = "#2B2B2B"
BLUE, ORANGE, PURPLE, GREEN, GREY = "#4C72B0", "#DD8452", "#8172B3", "#55A868", "#8C8C8C"


def detectability(evs):
    y = np.array([1 if e["is_ext"] else 0 for e in evs])
    prev = float(y.mean())
    out = {"n": len(evs), "prevalence": prev, "n_pos": int(y.sum()),
           "seeds": sorted({e["seed"] for e in evs})}
    for g, pref in GROUPS.items():
        haz, _ = off.oof_hazard(evs, pref)
        if np.ptp(haz) < 1e-9:
            out[g] = {"auprc": prev, "lift": 1.0, "roc": 0.5}
        else:
            ap = float(average_precision_score(y, haz))
            out[g] = {"auprc": ap, "lift": ap / prev if prev else float("nan"),
                      "roc": float(roc_auc_score(y, haz))}
    return out


def main():
    rows = {}
    for b in BENCHES:
        try:
            evs = e6.build_population([b], sample_live=0, seed=0)
        except Exception as ex:
            rows[b] = {"error": f"{type(ex).__name__}: {ex}"}
            continue
        if not evs or sum(e["is_ext"] for e in evs) < MIN_POS:
            npos = sum(e["is_ext"] for e in evs) if evs else 0
            rows[b] = {"n": len(evs), "n_pos": int(npos), "underpowered": True,
                       "note": f"< {MIN_POS} extinction events -> skipped OOF"}
            continue
        rows[b] = detectability(evs)

    res = {"min_pos": MIN_POS, "benchmarks": rows,
           "note": ("Layer 1 detectability only (Stage-A hazard AUPRC, no counterfactuals). "
                    "Necessary not sufficient: BBH shows lift yet the controller fails (E8). "
                    "Layer 2 (GPU) precomputes counterfactuals per benchmark for the full verdict.")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    # -------------------------------- print --------------------------------
    print("E11 Layer-1 detectability landscape (within-benchmark Stage-A hazard):")
    print(f"{'bench':14s} {'n':>4} {'prev':>5} {'npos':>4} | "
          f"{'ALL lift':>8} {'F1':>5} {'F3':>5} {'F4':>5} {'ALL ROC':>7}")
    plotb = []
    for b, r in rows.items():
        if r.get("error") or r.get("underpowered"):
            print(f"{b:14s}  -- {r.get('note') or r.get('error')}")
            continue
        print(f"{b:14s} {r['n']:>4} {r['prevalence']:>5.2f} {r['n_pos']:>4} | "
              f"{r['ALL']['lift']:>8.2f} {r['F1']['lift']:>5.2f} {r['F3']['lift']:>5.2f} "
              f"{r['F4']['lift']:>5.2f} {r['ALL']['roc']:>7.3f}")
        plotb.append(b)

    # -------------------------------- figure -------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#C7CFD8", "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(plotb)); w = 0.26
    for j, (g, col) in enumerate([("ALL", BLUE), ("F1", ORANGE), ("F3", PURPLE)]):
        ax.bar(x + (j - 1) * w, [rows[b][g]["lift"] for b in plotb], w,
               color=col, edgecolor=INK, linewidth=0.4, label=g)
    ax.axhline(1.0, ls="--", lw=1.2, color=GREY)
    ax.text(len(plotb) - 0.5, 1.02, "random (lift=1)", ha="right", fontsize=8, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={rows[b]['n']}, {rows[b]['prevalence']:.0%})" for b in plotb],
                       fontsize=8.5)
    ax.set_ylabel("Stage-A hazard AUPRC lift (within benchmark)")
    ax.set_title("E11 Layer 1 — where is extinction predictable? (necessary condition)\n"
                 "detectability ≠ controller success (BBH has lift but the controller fails, E8)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"\n[OK] {OUT}\n[OK] {FIG}")
    print("\nLAYER 2 (GPU, after per-seed E6): per-benchmark counterfactual precompute via\n"
          "  e6_controller.py --bench <b> --config config/config_e6_gpu1.yaml --cache <b>.pkl\n"
          "then the E6/E7/E9 offline verdict per benchmark -> cross-benchmark controller table.")


if __name__ == "__main__":
    main()
