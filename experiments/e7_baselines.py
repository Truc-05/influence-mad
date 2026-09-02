"""
E7 — Controller baselines (ZERO-GPU). Answers the reviewer's first question:
"why a learned hazard detector? why not just intervene when agents disagree?"

Every policy is a per-debate risk score; at a fixed intervention budget r we apply
the combined safeguard to the top-r fraction (matched budget). Because the round-0
structural heuristics are coarse (|A|in{1,2,3} -> f1 scores take ~3 values), a naive
top-k would break ties by array order and unfairly advantage/penalise them; we
therefore break ties at random and average over many draws. The random baseline is
just the constant-score policy under the same machinery.

Policies (all "higher score = intervene first"):
  random            constant score            -> matched-budget null (with CI)
  disagreement      f1_disagree_ratio         -> intervene where agents disagree
  entropy           f1_entropy                -> intervene at high answer entropy
  margin            -f1_support_margin        -> intervene at small majority margin
                                                 (= "low consensus strength"; the
                                                  deployable proxy for low confidence,
                                                  which the logs do not record)
  revision          f2_revision_count         -> DEGENERATE at gate g=0 (no agent has
                                                 revised yet); shown to coincide with
                                                 random -- revisions come too late.
  influence-only    OOF hazard on F3 only     -> PageRank/stubbornness/M_ij block
  FailSafe hazard   OOF hazard on all groups  -> the E6 controller (F1+F3+F4 live)

Output: results/tables/e7_baselines.json , results/figures/e7_baselines.png
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

OUT = "results/tables/e7_baselines.json"
FIG = "results/figures/e7_baselines.png"
BUDGETS = [0.20, 0.25, 0.30, 0.35, 0.40]
CURVE_R = np.round(np.arange(0.05, 0.601, 0.05), 3)
INK = "#2B2B2B"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, BROWN = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C", "#8172B3", "#937860")


acc_at_budget = off.acc_at_budget      # shared, validated matched-budget primitive


def main():
    events = off.load_events()
    yc, yk = off.outcome_vectors(events)
    y_ext = np.array([1 if e["is_ext"] else 0 for e in events])
    never, always = float(yc.mean()), float(yk.mean())
    rng = np.random.default_rng(0)

    F = lambda key: np.array([e["feat"][key] for e in events], float)
    haz_all, _ = off.oof_hazard(events, ("f1_", "f2_", "f3_", "f4_"))
    haz_f3, _ = off.oof_hazard(events, ("f3_",))

    policies = {
        "random":         (np.zeros(len(events)),        GREY,   "-",  1.6),
        "revision@gate":  (F("f2_revision_count"),       GREY,   ":",  1.6),
        "disagreement":   (F("f1_disagree_ratio"),       ORANGE, "-",  1.8),
        "entropy":        (F("f1_entropy"),               GREEN,  "-",  1.8),
        "margin(consensus)": (-F("f1_support_margin"),    BROWN,  "-",  1.8),
        "influence-only(F3)": (haz_f3,                    PURPLE, "-",  1.8),
        "FailSafe hazard": (haz_all,                      BLUE,   "-",  2.6),
    }

    # detector quality: how well each score ranks true extinctions
    ranking = {}
    for name, (s, *_ ) in policies.items():
        if np.ptp(s) < 1e-12:
            ranking[name] = {"auprc": float(y_ext.mean()), "roc": 0.5}
        else:
            ranking[name] = {"auprc": float(average_precision_score(y_ext, s)),
                             "roc": float(roc_auc_score(y_ext, s))}

    # matched-budget accuracy table + full curve
    table = {name: {} for name in policies}
    curve = {name: {"r": [], "acc": [], "lo": [], "hi": []} for name in policies}
    for name, (s, *_ ) in policies.items():
        for b in BUDGETS:
            m, lo, hi = acc_at_budget(s, yc, yk, b, rng)
            table[name][f"{b:.2f}"] = {"acc": m, "lo": lo, "hi": hi,
                                       "gr": off.gain_retention(m, never, always)}
        for b in CURVE_R:
            m, lo, hi = acc_at_budget(s, yc, yk, b, rng)
            curve[name]["r"].append(float(b)); curve[name]["acc"].append(m)
            curve[name]["lo"].append(lo); curve[name]["hi"].append(hi)

    res = {"n": len(events), "n_questions": len({(e["bench"], e["qid"]) for e in events}),
           "never_acc": never, "always_acc": always, "prevalence_ext": float(y_ext.mean()),
           "budgets": BUDGETS, "ranking_of_extinction": ranking,
           "matched_budget": table, "curve": curve,
           "note": "revision@gate is degenerate (F2==0 at g=0) and coincides with random."}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    # ------------------------------- print ---------------------------------
    print(f"E7 baselines  n={res['n']} ({res['n_questions']} questions)  "
          f"never={never:.3f} always={always:.3f}")
    print(f"\n{'policy':20s} {'AUPRC':>6} {'ROC':>5} | " +
          "  ".join(f"r={b:.0%}" for b in BUDGETS))
    for name in policies:
        row = "  ".join(f"{table[name][f'{b:.2f}']['acc']:.3f}" for b in BUDGETS)
        print(f"{name:20s} {ranking[name]['auprc']:>6.3f} {ranking[name]['roc']:>5.2f} | {row}")
    # headline gaps at 30%
    h30 = table["FailSafe hazard"]["0.30"]["acc"]
    best_heur = max(("disagreement", "entropy", "margin(consensus)"),
                    key=lambda k: table[k]["0.30"]["acc"])
    print(f"\n@30% budget: FailSafe {h30:.3f}  vs best heuristic "
          f"({best_heur}) {table[best_heur]['0.30']['acc']:.3f}  vs "
          f"influence-only {table['influence-only(F3)']['0.30']['acc']:.3f}  vs "
          f"random {table['random']['0.30']['acc']:.3f}")

    # ------------------------------- figure --------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#C7CFD8", "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.axvspan(0.20, 0.40, color=ORANGE, alpha=0.08, lw=0)
    ax.axhline(always, ls="--", lw=1.1, color="#55A868")
    ax.axhline(never, ls="--", lw=1.1, color="#C44E52")
    ax.text(0.052, always + 0.001, f"always-on {always:.1%}", color="#55A868", fontsize=8, va="bottom")
    ax.text(0.052, never + 0.001, f"never {never:.1%}", color="#C44E52", fontsize=8, va="bottom")
    ax.text(0.30, always - 0.004, "20–40% operating band", ha="center", va="top",
            fontsize=8, color="#9A6A2E")

    order_draw = ["random", "revision@gate", "margin(consensus)", "entropy",
                  "disagreement", "influence-only(F3)", "FailSafe hazard"]
    for name in order_draw:
        s, col, style, lw = policies[name]
        c = curve[name]
        if name == "random":
            ax.fill_between(c["r"], c["lo"], c["hi"], color=GREY, alpha=0.18, lw=0)
        lbl = name + (" (≡ random)" if name == "revision@gate" else "")
        ax.plot(c["r"], c["acc"], color=col, ls=style, lw=lw, marker="o", ms=3.4,
                label=lbl, zorder=5 if name == "FailSafe hazard" else 3)

    ax.set_xlabel("Intervention budget  (fraction of debates given the combined safeguard)")
    ax.set_ylabel("Debate accuracy")
    ax.set_title("E7 — the learned hazard is the best controller signal;\n"
                 "disagreement geometry drives it, influence alone does not",
                 fontsize=10.5)
    ax.set_xlim(0.04, 0.61)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", fontsize=8.2, ncol=1)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIG, dpi=300, facecolor="white"); plt.close(fig)
    print(f"\n[OK] {OUT}\n[OK] {FIG}")


if __name__ == "__main__":
    main()
