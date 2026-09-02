"""
E6 figures — hazard-guided controller (reads the frozen results table, no GPU).

Main figure (2 panels), house seaborn-muted style to match E1/E4a:
  Panel A  Accuracy vs intervention rate: hazard-guided controller, random-matched
           budget (95% CI band), never floor, always-on ceiling, Pareto star @ tau*.
  Panel B  Gain Retention vs intervention rate: hazard vs random-matched, 20-40%
           operating band shaded, GR>=0.70 gate line, Pareto star.

Appendix figure: collateral asymmetry — rescue on would-die vs harm on would-live.

Usage:  python experiments/e6_figure.py
Outputs: results/figures/e6_controller.png , results/figures/e6_collateral.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results/tables/e6_controller.json"
FIG_DIR = "results/figures"

INK = "#2B2B2B"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C", "#8172B3")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#C7CFD8", "axes.linewidth": 0.9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "legend.frameon": False,
})


def load():
    d = json.load(open(SRC))
    # sort both curves by ascending intervention rate for clean lines
    hz = sorted(d["hazard_curve"], key=lambda p: p["rate"])
    rd = sorted(d["random_curve"], key=lambda p: p["rate"])
    return d, hz, rd


def gr_of(acc, never, always):
    return (acc - never) / (always - never)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    d, hz, rd = load()
    never, always = d["never_acc"], d["always_acc"]
    n = d["n"]

    hr = np.array([p["rate"] for p in hz]); ha = np.array([p["acc"] for p in hz])
    hgr = np.array([p["gain_retention"] for p in hz])
    rr = np.array([p["rate"] for p in rd]); ra = np.array([p["acc"] for p in rd])
    rlo = np.array([p["acc_ci"][0] for p in rd]); rhi = np.array([p["acc_ci"][1] for p in rd])
    rgr = np.array([gr_of(a, never, always) for a in ra])
    rgr_lo = np.array([gr_of(a, never, always) for a in rlo])
    rgr_hi = np.array([gr_of(a, never, always) for a in rhi])

    # Pareto operating point = hazard point of maximum accuracy (observed, not tuned)
    star = max(hz, key=lambda p: p["acc"])
    s_rate, s_acc, s_tau, s_gr = star["rate"], star["acc"], star["tau"], star["gain_retention"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # -------------------------------- Panel A --------------------------------
    # achievable-gain corridor between the never floor and always ceiling
    axA.axhspan(never, always, color=GREEN, alpha=0.06, zorder=0)
    axA.axhline(always, ls="--", lw=1.3, color=GREEN, zorder=1)
    axA.axhline(never, ls="--", lw=1.3, color=RED, zorder=1)
    axA.text(0.015, always + 0.0015, f"always-on ceiling  {always:.1%}",
             color=GREEN, fontsize=8.5, va="bottom")
    axA.text(0.015, never - 0.002, f"never (no intervention)  {never:.1%}",
             color=RED, fontsize=8.5, va="top")

    axA.fill_between(rr, rlo, rhi, color=GREY, alpha=0.18, lw=0, zorder=2)
    axA.plot(rr, ra, color=GREY, lw=1.8, marker="o", ms=3.2,
             label="random-matched budget (95% CI)", zorder=3)
    axA.plot(hr, ha, color=BLUE, lw=2.4, marker="o", ms=4.2,
             label="hazard-guided controller", zorder=4)

    axA.scatter([s_rate], [s_acc], s=180, marker="*", color=ORANGE,
                edgecolor=INK, linewidth=0.8, zorder=6)
    axA.annotate(f"$\\tau^\\star={s_tau:.2f}$ (observed Pareto point)\n"
                 f"{s_acc:.1%} at {s_rate:.1%} of interventions\n"
                 f"$\\gtrsim$ always-on ({always:.1%}), fewer interventions",
                 xy=(s_rate, s_acc), xytext=(0.40, 0.578),
                 fontsize=8.3, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                                 connectionstyle="arc3,rad=0.22"))

    axA.set_xlim(-0.02, 1.02); axA.set_ylim(0.560, 0.672)
    axA.set_xlabel("Intervention rate  (fraction of debates given the combined safeguard)")
    axA.set_ylabel("Debate accuracy")
    axA.set_title("A  Selective beats indiscriminate intervention")
    axA.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axA.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axA.legend(loc="lower right"); axA.grid(True, alpha=0.28)

    # -------------------------------- Panel B --------------------------------
    axB.axvspan(0.20, 0.40, color=ORANGE, alpha=0.10, lw=0, zorder=0)
    axB.text(0.30, 1.11, "20–40%\noperating band", ha="center", va="top",
             fontsize=8, color="#9A6A2E")
    axB.axhline(1.0, ls="--", lw=1.2, color=GREEN, zorder=1)
    axB.axhline(0.0, ls="--", lw=1.0, color=RED, zorder=1)
    axB.axhline(0.70, ls=":", lw=1.1, color=GREY, zorder=1)
    axB.text(0.985, 0.71, "GR = 0.70 gate", ha="right", va="bottom",
             fontsize=7.8, color=GREY)

    axB.fill_between(rr, rgr_lo, rgr_hi, color=GREY, alpha=0.18, lw=0, zorder=2)
    axB.plot(rr, rgr, color=GREY, lw=1.8, marker="o", ms=3.2,
             label="random-matched budget", zorder=3)
    axB.plot(hr, hgr, color=BLUE, lw=2.4, marker="o", ms=4.2,
             label="hazard-guided controller", zorder=4)
    axB.scatter([s_rate], [s_gr], s=180, marker="*", color=ORANGE,
                edgecolor=INK, linewidth=0.8, zorder=6)

    axB.set_xlim(-0.02, 1.02); axB.set_ylim(-0.05, 1.18)
    axB.set_xlabel("Intervention rate")
    axB.set_ylabel("Gain retention  (Acc − never) / (always − never)")
    axB.set_title("B  Detector value = gap over matched-random budget")
    axB.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axB.legend(loc="lower right"); axB.grid(True, alpha=0.28)

    fig.suptitle(
        f"E6 — hazard-guided adaptive controller   (n = {n} debates with a correct "
        f"answer alive at round 0)", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f"{FIG_DIR}/e6_controller.png"
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    print(f"[OK] {out}")

    # ------------------------- appendix: collateral --------------------------
    c = d["collateral"]
    help_, harm = c["help_rate_on_would_die"], c["harm_rate_on_would_live"]
    n_wd, n_wl = c["n_would_die"], c["n_would_live"]

    figc, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = [("Rescued\n(would-die debates)", help_, GREEN, n_wd),
            ("Collateral harm\n(would-live debates)", -harm, RED, n_wl)]
    ypos = [1, 0]
    for (lab, val, col, nn), y in zip(bars, ypos):
        ax.barh(y, val, height=0.52, color=col, alpha=0.9, edgecolor=INK, linewidth=0.6)
        ax.text(val + (0.02 if val > 0 else -0.02), y,
                f"{abs(val):.1%}", va="center",
                ha="left" if val > 0 else "right", fontsize=11, fontweight="bold", color=col)
        ax.text(0.0, y + 0.34, f"{lab}   (n={nn})", va="bottom",
                ha="left" if val > 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks([]); ax.set_ylim(-0.6, 1.7); ax.set_xlim(-0.42, 0.62)
    ax.set_xlabel("Effect of always applying the combined safeguard")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    ax.set_title("Intervention is asymmetric — the controller's reason to exist", fontsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25)
    figc.tight_layout()
    outc = f"{FIG_DIR}/e6_collateral.png"
    figc.savefig(outc, dpi=300, facecolor="white"); plt.close(figc)
    print(f"[OK] {outc}")


if __name__ == "__main__":
    main()
