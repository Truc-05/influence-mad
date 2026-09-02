"""
E8 — Ablation of the hazard-guided controller (ZERO-GPU). Two questions:
  (1) which feature groups create the gain?           -> pooled group ablation
  (2) is the gain a real per-task signal or a pooling
      artifact of gsm8k vs bbh differing in prevalence
      and verbosity?                                  -> within-benchmark validation

Honest findings (do not overclaim a single "workhorse"):
  * The gain is REAL within gsm8k (ALL: AUPRC lift ~2x, gain-retention@40% ~0.94) --
    not merely benchmark separation.
  * It does NOT transfer to bbh: within bbh every feature-group controller sits at or
    below random at 40% budget. The pooled result is gsm8k-driven.
  * No single channel dominates across tasks: influence (F3) is the best ranker inside
    gsm8k yet ~random inside bbh; text (F4) carries the most *pooled* signal (partly a
    verbosity confound). The combined detector (ALL) is the robust choice.
  * At the g=0 gate the F2 dynamics block is identically zero (nothing has revised
    yet); the pooled multi-time detector (e3_router.json), where F2 is alive, is
    printed for contrast.

Output: results/tables/e8_ablation.json , results/figures/e8_ablation.png
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

import experiments.e6_offline as off

OUT = "results/tables/e8_ablation.json"
FIG = "results/figures/e8_ablation.png"
INK = "#2B2B2B"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C", "#8172B3")

POOLED_VARIANTS = {                      # name -> prefixes (F2 dead at g=0, omitted)
    "F3 only (influence)": ("f3_",),
    "F4 only (text)":      ("f4_",),
    "F3+F4  (−structure)": ("f3_", "f4_"),
    "F1 only (structure)": ("f1_",),
    "F1+F4  (−influence)": ("f1_", "f4_"),
    "F1+F3  (−text)":      ("f1_", "f3_"),
    "ALL  (F1+F3+F4)":     ("f1_", "f3_", "f4_"),
}
CORE = {"F1": ("f1_",), "F3": ("f3_",), "F4": ("f4_",), "ALL": ("f1_", "f2_", "f3_", "f4_")}


def evalset(evs, prefixes, rng):
    yc, yk = off.outcome_vectors(evs)
    never, always = float(yc.mean()), float(yk.mean())
    y = np.array([1 if e["is_ext"] else 0 for e in evs])
    haz, keys = off.oof_hazard(evs, prefixes)
    a30 = off.acc_at_budget(haz, yc, yk, 0.30, rng)[0]
    a40 = off.acc_at_budget(haz, yc, yk, 0.40, rng)[0]
    rnd40 = off.acc_at_budget(np.zeros(len(evs)), yc, yk, 0.40, rng)[0]
    prev = float(y.mean())
    return {"n": len(evs), "n_feats": len(keys), "prevalence": prev,
            "auprc": float(average_precision_score(y, haz)),
            "lift": float(average_precision_score(y, haz) / prev) if prev else float("nan"),
            "roc": float(roc_auc_score(y, haz)),
            "acc@30": a30, "gr@30": off.gain_retention(a30, never, always),
            "acc@40": a40, "gr@40": off.gain_retention(a40, never, always),
            "rand_acc@40": rnd40, "edge@40": a40 - rnd40,   # raw acc edge over random (stable)
            "random_gr@40": off.gain_retention(rnd40, never, always),
            "auc_2040": off.auc_band(haz, yc, yk, rng)}


def main():
    events = off.load_events()
    rng = np.random.default_rng(0)

    # (1) pooled feature-group ablation
    pooled = {name: evalset(events, pref, rng) for name, pref in POOLED_VARIANTS.items()}
    A = pooled["ALL  (F1+F3+F4)"]
    loo = {
        "drop_structure(F1)": {"d_auprc": A["auprc"] - pooled["F3+F4  (−structure)"]["auprc"],
                               "d_gr40": A["gr@40"] - pooled["F3+F4  (−structure)"]["gr@40"]},
        "drop_influence(F3)": {"d_auprc": A["auprc"] - pooled["F1+F4  (−influence)"]["auprc"],
                               "d_gr40": A["gr@40"] - pooled["F1+F4  (−influence)"]["gr@40"]},
        "drop_text(F4)":      {"d_auprc": A["auprc"] - pooled["F1+F3  (−text)"]["auprc"],
                               "d_gr40": A["gr@40"] - pooled["F1+F3  (−text)"]["gr@40"]},
    }

    # (2) within-benchmark validation for the core groups
    scopes = {"gsm8k": [e for e in events if e["bench"] == "gsm8k"],
              "bbh":   [e for e in events if e["bench"] == "bbh"],
              "pooled": events}
    perbench = {sc: {g: evalset(evs, pref, rng) for g, pref in CORE.items()}
                for sc, evs in scopes.items()}

    # pooled multi-time detector (context; F2 alive there)
    e3pool = {}
    try:
        e3 = json.load(open("results/tables/e3_router.json"))["stageA"]["pooled"]
        for k in ("F1|lr", "F1+F2|lr", "F1+F2+F3|lr", "ALL|lr"):
            if k in e3:
                e3pool[k] = {"auprc": e3[k]["auprc"], "roc": e3[k]["roc"]}
    except Exception as ex:
        e3pool = {"error": str(ex)}

    res = {"gate_note": "F2==0 at g=0; ablation over live groups F1/F3/F4",
           "pooled_ablation": pooled, "leave_one_out": loo,
           "within_benchmark": perbench, "pooled_multitime_detector_e3a": e3pool,
           "conclusion": ("with per-seed outcomes, the detector still RANKS extinctions "
                          "(gsm8k ALL AUPRC lift ~2x) but the accuracy edge over a random "
                          "budget @40% is ~0 within gsm8k and bbh (only pooled marginally "
                          "positive); the earlier 'gsm8k works' was inflated by the "
                          "seed-collapse. No single channel dominates; ALL is robust.")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    # -------------------------------- print --------------------------------
    print("E8  (1) POOLED feature-group ablation (controller gate g=0):")
    print(f"{'variant':22s} {'AUPRC':>6} {'lift':>5} {'acc@40':>7} {'GR@40':>6} {'AUC20-40':>9}")
    for n, r in pooled.items():
        print(f"{n:22s} {r['auprc']:>6.3f} {r['lift']:>5.2f} {r['acc@40']:>7.3f} "
              f"{r['gr@40']:>6.2f} {r['auc_2040']:>9.3f}")
    print("  leave-one-out from ALL:  " + "  ".join(
        f"{k}: ΔAUPRC={v['d_auprc']:+.3f} ΔGR@40={v['d_gr40']:+.2f}" for k, v in loo.items()))

    print("\nE8  (2) WITHIN-BENCHMARK validation (AUPRC lift / accuracy edge over random @40%):")
    print(f"{'scope':8s} {'n':>4} {'prev':>5} | " +
          "  ".join(f"{g}:lift/edge40" for g in CORE))
    for sc, gs in perbench.items():
        cells = "  ".join(f"{gs[g]['lift']:.2f}/{gs[g]['edge@40']:+.3f}" for g in CORE)
        print(f"{sc:8s} {gs['ALL']['n']:>4} {gs['ALL']['prevalence']:>5.2f} | {cells}")
    print("  -> detector RANKS extinctions (lift>1) but the accuracy edge over a random "
          "budget is ~0 on both benchmarks; only pooled is marginally positive.")

    # -------------------------------- figure -------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#C7CFD8", "legend.frameon": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9))

    # Panel A: pooled ablation AUPRC (horizontal bars)
    names = list(POOLED_VARIANTS.keys()); yp = np.arange(len(names))
    def col(nm):
        return BLUE if nm.startswith("ALL") else (
            ORANGE if nm.startswith("F4 only") else (
            RED if "−structure" in nm else "#B7BEC7"))
    ax1.barh(yp, [pooled[n]["auprc"] for n in names],
             color=[col(n) for n in names], edgecolor=INK, linewidth=0.5)
    prev = pooled["ALL  (F1+F3+F4)"]["prevalence"]
    ax1.axvline(prev, ls="--", lw=1.0, color=GREY)
    ax1.text(prev, len(names) - 0.35, " random\n (prevalence)", fontsize=7.3, color=GREY, va="top")
    for i, n in enumerate(names):
        ax1.text(pooled[n]["auprc"] + 0.004, i, f"{pooled[n]['auprc']:.3f}", va="center", fontsize=8)
    ax1.set_yticks(yp); ax1.set_yticklabels(names, fontsize=8.8)
    ax1.set_xlabel("Detector AUPRC (ranking extinctions)")
    ax1.set_title("A  Pooled ablation: no single channel dominates,\n"
                  "influence (F3) is not required", fontsize=10)
    ax1.set_xlim(0, max(pooled[n]["auprc"] for n in names) * 1.2)
    ax1.grid(True, axis="x", alpha=0.28)

    # Panel B: within-benchmark accuracy EDGE over a random budget @40% (stable metric;
    # gain-retention is unreliable now that the always-never headroom is only ~4.5pp)
    scope_names = ["gsm8k", "bbh", "pooled"]; groups = ["F1", "F3", "F4", "ALL"]
    gcol = {"F1": ORANGE, "F3": PURPLE, "F4": GREEN, "ALL": BLUE}
    x = np.arange(len(scope_names)); w = 0.2
    for j, g in enumerate(groups):
        ax2.bar(x + (j - 1.5) * w, [perbench[s][g]["edge@40"] for s in scope_names],
                w, color=gcol[g], edgecolor=INK, linewidth=0.4, label=g)
    ax2.axhline(0, color=INK, lw=1.0)
    ax2.set_xticks(x); ax2.set_xticklabels(
        [f"{s}\n(n={perbench[s]['ALL']['n']})" for s in scope_names])
    ax2.set_ylabel("Accuracy edge over random budget @ 40%")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax2.set_title("B  Edge over a random budget is small everywhere\n"
                  "(the seed-collapse had inflated gsm8k)", fontsize=10)
    ax2.legend(fontsize=8, ncol=2, loc="upper right")
    ax2.grid(True, axis="y", alpha=0.28)

    fig.suptitle("E8 — the detector ranks extinctions (A) but targeting them barely beats a "
                 "random budget (B); influence is benchmark-dependent, not the core",
                 fontsize=10.3, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"\n[OK] {OUT}\n[OK] {FIG}")


if __name__ == "__main__":
    main()
