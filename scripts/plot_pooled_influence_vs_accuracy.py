"""Redraw results/figures/pooled_influence_vs_accuracy.png as a dumbbell chart.

Ý tưởng: influence (PageRank) và accuracy khác đơn vị nên không so trực tiếp
được. Chuẩn hoá cả hai thành *share trong nội bộ benchmark*:

    influence_share = PageRank_a            (đã tự sum = 1 trên 3 agent)
    accuracy_share  = acc_a / sum(acc)      (agent này đóng góp bao nhiêu phần
                                             trong tổng số câu đúng của nhóm)

Mỗi agent = một thanh nối 2 chấm. Chấm influence nằm bên phải chấm accuracy
=> agent được lắng nghe nhiều hơn mức nó xứng đáng (over-influential), và
ngược lại. Đường đứt ở 1/3 là mốc "chia đều 3 agent".
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "tables", "analysis_findings.json")
FIG = os.path.join(ROOT, "results", "figures")
OUT = os.path.join(FIG, "pooled_influence_vs_accuracy.png")           # paper
OUT_TITLED = os.path.join(FIG, "pooled_influence_vs_accuracy_titled.png")  # slide

BENCH_LABEL = {
    "gsm8k": "GSM8K",
    "strategyqa": "StrategyQA",
    "commonsenseqa": "CommonsenseQA",
    "bbh": "BBH",
    "truthfulqa": "TruthfulQA",
}
AGENT_ORDER = ["careful", "skeptic", "bigpicture"]


C_INF = "#C0392B"   # influence share
C_ACC = "#2E6DA4"   # accuracy share
C_BAR = "#CBD3DB"
INK = "#1F2933"
MUTED = "#6B7683"


def render(paper_style, out):
    with open(SRC, encoding="utf-8") as f:
        master = json.load(f)["benchmarks"]
    benches = [b for b in BENCH_LABEL if b in master] or list(master)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "xtick.labelsize": 10,
            "ytick.color": INK,
            "axes.edgecolor": "#C7CFD8",
        }
    )

    fig, axes = plt.subplots(
        len(benches), 1, figsize=(7.0, 7.3), sharex=True,
        gridspec_kw={"hspace": 0.62},
    )
    if len(benches) == 1:
        axes = [axes]

    all_vals = []
    for ax, b in zip(axes, benches):
        m = master[b]
        agents = [a for a in AGENT_ORDER if a in m["influence"]]
        acc_tot = sum(m["accuracy"][a] for a in agents)
        rows = []
        for a in agents:
            rows.append((a, m["influence"][a], m["accuracy"][a] / acc_tot))
        all_vals += [v for r in rows for v in r[1:]]

        ys = np.arange(len(rows))[::-1]  # agent đầu tiên nằm trên cùng
        for y, (a, inf, acc) in zip(ys, rows):
            ax.plot([acc, inf], [y, y], "-", color=C_BAR, lw=6.2,
                    solid_capstyle="round", zorder=1)
            ax.scatter(acc, y, s=115, color=C_ACC, zorder=3,
                       edgecolor="white", linewidth=1.2)
            ax.scatter(inf, y, s=115, color=C_INF, zorder=3,
                       edgecolor="white", linewidth=1.2)
            delta = inf - acc
            ax.text(
                max(inf, acc) + 0.013, y,
                f"{delta * 100:+.1f} pp",
                va="center", ha="left", fontsize=9.5, fontweight="bold",
                color=C_INF if delta > 0 else C_ACC,
            )

        ax.axvline(1 / 3, ls=(0, (4, 3)), lw=1.1, color="#9AA5B1", zorder=0)
        # KHÔNG in `pearson_inf_acc` lên panel: giá trị đó là trung bình của r
        # từng seed, còn các chấm ở đây là trung bình seed. Hai ước lượng khác
        # nhau (StrategyQA: 0.24 vs 0.93), in lên sẽ mâu thuẫn với chính hình.
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=10)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(
            BENCH_LABEL.get(b, b), fontsize=11, fontweight="bold",
            loc="left", pad=9, x=-0.175,
        )
        ax.grid(True, axis="x", color="#EDF1F5", lw=1.0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)

    lo, hi = min(all_vals), max(all_vals)
    axes[0].set_xlim(lo - 0.045, hi + 0.105)
    axes[-1].set_xlabel(
        "Share within benchmark", fontsize=11, fontweight="bold", labelpad=10,
    )
    axes[-1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].annotate(
        "equal split (33%)",
        xy=(1 / 3, len(AGENT_ORDER) - 0.45), xytext=(6, 0),
        textcoords="offset points", ha="left", va="center",
        fontsize=9, color=MUTED,
    )

    if not paper_style:
        fig.suptitle(
            "Influence is not earned by being right",
            x=0.055, y=0.993, fontsize=15.5, fontweight="bold", ha="left",
            va="top",
        )
        fig.text(
            0.055, 0.947,
            "Red dot right of blue = the agent is listened to more than its "
            "accuracy justifies",
            ha="left", fontsize=10.5, color=MUTED,
        )

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=10, color=C_INF,
                   label="Influence share (PageRank)"),
        plt.Line2D([], [], marker="o", ls="", ms=10, color=C_ACC,
                   label="Share of correct answers"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False,
        fontsize=10, bbox_to_anchor=(0.55, 0.012),
    )

    fig.subplots_adjust(
        left=0.20, right=0.985, top=0.945 if paper_style else 0.885, bottom=0.145
    )
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    render(paper_style=True, out=OUT)          # IEEE: caption gánh tiêu đề
    render(paper_style=False, out=OUT_TITLED)  # slide/README: có tiêu đề
