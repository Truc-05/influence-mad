"""
Phân tích sâu, tổng hợp qua tất cả benchmark + seed cho định vị ANALYSIS:
  1. Bảng influence/accuracy/role per agent per benchmark (mean±std qua seed).
  2. Tính ổn định thứ hạng influence qua các benchmark (Spearman).
  3. Nguồn gốc influence: tương quan influence với (a) accuracy độc lập vòng 0,
     (b) mức "bướng" (1 - tần suất đổi ý). Influence đến từ đúng hay từ lì?
  4. Error-cascade: debate biến câu 'thắng được' thành thua, và ngược lại.
  5. Voting: majority vs influence_based vs influence_aware.
Xuất: results/tables/analysis_findings.json + results/figures/*.png
"""
import json, glob, os, sys, statistics
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.utils.io_utils import read_jsonl
from src.utils.scoring import is_correct

BENCH_ORDER = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa"]
FIG_DIR = "results/figures"; TAB_DIR = "results/tables"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)


def _norm(t): return (t or "").strip().lower()


def load_multiseed(bench):
    p = f"results/tables/{bench}_multiseed.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def seed_logs(bench):
    out = {}
    for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
        out[f] = read_jsonl(f)
    return out


def role(inf, acc, agents):
    mi = float(np.median([inf[a] for a in agents]))
    ma = float(np.median([acc[a] for a in agents]))
    r = {}
    for a in agents:
        hi, ha = inf[a] >= mi, acc[a] >= ma
        r[a] = "good" if (hi and ha) else "bad" if (hi and not ha) else "low"
    return r


def independent_and_stubbornness(bench, agents):
    """acc vòng 0 (độc lập) + độ bướng (1 - tần suất đổi ý) trung bình qua seed."""
    ind_correct = {a: 0 for a in agents}; ind_total = {a: 0 for a in agents}
    kept = {a: 0 for a in agents}; opp = {a: 0 for a in agents}
    for _, logs in seed_logs(bench).items():
        for lg in logs:
            gt = lg["ground_truth"]; ca = lg.get("correct_answers")
            ind = lg.get("independent_answers", {})
            for a in agents:
                if a in ind:
                    ind_total[a] += 1
                    if is_correct(ind[a], gt, bench, ca): ind_correct[a] += 1
            for rd in lg["rounds"][1:]:
                ch = rd.get("changed", {})
                for a in agents:
                    if a in ch:
                        opp[a] += 1
                        if not ch[a]: kept[a] += 1
    ind_acc = {a: (ind_correct[a]/ind_total[a] if ind_total[a] else 0.0) for a in agents}
    stubborn = {a: (kept[a]/opp[a] if opp[a] else 0.0) for a in agents}
    return ind_acc, stubborn


def error_cascade(bench, agents):
    """Theo từng câu (gộp mọi seed) dùng majority của đáp án CUỐI vs độc lập.
    broke: >=1 agent đúng ở vòng 0 nhưng majority cuối SAI.
    fixed: majority độc lập SAI nhưng majority cuối ĐÚNG."""
    def maj(ans):  # ans: dict agent->answer
        vals = [_norm(x) for x in ans.values()]
        key = max(set(vals), key=vals.count)
        # trả về 1 answer gốc khớp key
        for a, x in ans.items():
            if _norm(x) == key: return x
        return list(ans.values())[0]
    broke = fixed = total = 0
    any_correct_lost = 0
    for _, logs in seed_logs(bench).items():
        for lg in logs:
            gt = lg["ground_truth"]; ca = lg.get("correct_answers")
            ind = lg.get("independent_answers", {}); fin = lg.get("final_answers", {})
            if not ind or not fin: continue
            total += 1
            ind_any_correct = any(is_correct(ind[a], gt, bench, ca) for a in ind)
            ind_maj_correct = is_correct(maj(ind), gt, bench, ca)
            fin_maj_correct = is_correct(maj(fin), gt, bench, ca)
            fin_any_correct = any(is_correct(fin[a], gt, bench, ca) for a in fin)
            if ind_maj_correct and not fin_maj_correct: broke += 1
            if (not ind_maj_correct) and fin_maj_correct: fixed += 1
            if ind_any_correct and not fin_any_correct: any_correct_lost += 1
    return {"total": total, "broke": broke, "fixed": fixed, "net_fixed": fixed - broke,
            "consensus_to_wrong": any_correct_lost}


def main():
    benches = [b for b in BENCH_ORDER if load_multiseed(b)]
    master = {}
    pooled_inf, pooled_acc = [], []      # cho scatter gộp
    inf_rank_vectors = {}                # bench -> [inf per agent] để xét stability
    src_rows = []                        # (agent,bench,inf,ind_acc,stubborn)

    for b in benches:
        d = load_multiseed(b); agents = d["agent_names"]
        inf = {a: d["influence_scores"][a]["mean"] for a in agents}
        acc = {a: d["accuracy_per_agent"][a]["mean"] for a in agents}
        rl = role(inf, acc, agents)
        ind_acc, stubborn = independent_and_stubbornness(b, agents)
        casc = error_cascade(b, agents)
        corrs = [s["correlation"].get("pearson_r") for s in d["per_seed"]
                 if s["correlation"].get("pearson_r") is not None]
        pear = statistics.mean(corrs) if corrs else None

        master[b] = {"agents": agents, "influence": inf, "accuracy": acc, "role": rl,
                     "independent_acc": ind_acc, "stubbornness": stubborn,
                     "pearson_inf_acc": pear, "cascade": casc,
                     "voting": {k: d["voting_accuracy"][k] for k in
                                ["majority","influence_based","influence_aware"]}}
        inf_rank_vectors[b] = [inf[a] for a in agents]
        for a in agents:
            pooled_inf.append(inf[a]); pooled_acc.append(acc[a])
            src_rows.append((a, b, inf[a], ind_acc[a], stubborn[a]))

    # ---- stability: Spearman thứ hạng influence giữa mọi cặp benchmark ----
    stab = []
    for i in range(len(benches)):
        for j in range(i+1, len(benches)):
            r, _ = spearmanr(inf_rank_vectors[benches[i]], inf_rank_vectors[benches[j]])
            stab.append(r)
    stability = {"pairwise_spearman_mean": float(np.mean(stab)) if stab else None,
                 "pairwise_spearman_min": float(np.min(stab)) if stab else None}

    # ---- nguồn gốc influence (gộp 15 điểm agent×bench) ----
    infs = [r[2] for r in src_rows]; inds = [r[3] for r in src_rows]; stbs = [r[4] for r in src_rows]
    src = {}
    if len(set(infs)) > 1:
        r1, p1 = pearsonr(infs, inds); r2, p2 = pearsonr(infs, stbs)
        src = {"influence_vs_independent_acc": {"r": r1, "p": p1},
               "influence_vs_stubbornness": {"r": r2, "p": p2},
               "pooled_influence_vs_final_acc": dict(zip(["r","p"], pearsonr(pooled_inf, pooled_acc)))}

    findings = {"benchmarks": master, "stability": stability, "influence_source": src}
    json.dump(findings, open(f"{TAB_DIR}/analysis_findings.json","w",encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)

    # ============ FIGURES ============
    # (1) influence vs accuracy grouped bars per benchmark
    fig, axes = plt.subplots(1, len(benches), figsize=(3.2*len(benches), 3.4), sharey=True)
    if len(benches) == 1: axes = [axes]
    for ax, b in zip(axes, benches):
        agents = master[b]["agents"]; x = np.arange(len(agents)); w=0.38
        ax.bar(x-w/2, [master[b]["influence"][a] for a in agents], w, label="influence", color="#4C72B0")
        ax.bar(x+w/2, [master[b]["accuracy"][a] for a in agents], w, label="accuracy", color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels(agents, rotation=20, fontsize=8)
        ax.set_title(b, fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("score"); axes[0].legend(fontsize=8)
    fig.suptitle("Influence vs Accuracy per agent (mean over 5 seeds)")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/all_influence_vs_accuracy.png", dpi=150); plt.close(fig)

    # (2) scatter gộp influence vs final accuracy, màu theo benchmark
    fig, ax = plt.subplots(figsize=(5.5,4.2))
    cmap = plt.get_cmap("tab10")
    for k,b in enumerate(benches):
        agents = master[b]["agents"]
        xs=[master[b]["influence"][a] for a in agents]; ys=[master[b]["accuracy"][a] for a in agents]
        ax.scatter(xs, ys, color=cmap(k), label=b, s=60)
        for a,xx,yy in zip(agents,xs,ys):
            ax.annotate(a[:4], (xx,yy), fontsize=7, xytext=(4,3), textcoords="offset points")
    ax.set_xlabel("Influence score"); ax.set_ylabel("Accuracy")
    ax.set_title("Influence vs Accuracy (pooled)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/pooled_influence_vs_accuracy.png", dpi=150); plt.close(fig)

    # (3) voting comparison grouped bars
    fig, ax = plt.subplots(figsize=(7,4))
    x=np.arange(len(benches)); w=0.26
    for k,strat in enumerate(["majority","influence_based","influence_aware"]):
        means=[master[b]["voting"][strat]["mean"] for b in benches]
        errs=[master[b]["voting"][strat]["std"] for b in benches]
        ax.bar(x+(k-1)*w, means, w, yerr=errs, capsize=3, label=strat)
    ax.set_xticks(x); ax.set_xticklabels(benches, rotation=15); ax.set_ylabel("Accuracy")
    ax.set_title("Voting strategies (mean±std, 5 seeds)"); ax.legend(fontsize=8); ax.grid(True,axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/all_voting.png", dpi=150); plt.close(fig)

    # ============ PRINT SUMMARY ============
    print("\n#### ROLE per benchmark (influence / accuracy) ####")
    for b in benches:
        m=master[b]
        parts=[f"{a}:{m['role'][a]}(inf{m['influence'][a]:.2f}/acc{m['accuracy'][a]:.2f})" for a in m["agents"]]
        print(f"{b:14s} pear(inf,acc)={m['pearson_inf_acc']:.2f} | " + "  ".join(parts))
    print(f"\nStability of influence ranking across benchmarks: "
          f"Spearman mean={stability['pairwise_spearman_mean']:.3f} min={stability['pairwise_spearman_min']:.3f}")
    if src:
        print(f"Influence vs independent-acc: r={src['influence_vs_independent_acc']['r']:.3f} "
              f"(p={src['influence_vs_independent_acc']['p']:.3f})")
        print(f"Influence vs stubbornness  : r={src['influence_vs_stubbornness']['r']:.3f} "
              f"(p={src['influence_vs_stubbornness']['p']:.3f})")
    print("\n#### ERROR CASCADE (pooled over seeds) ####")
    for b in benches:
        c=master[b]["cascade"]
        print(f"{b:14s} total={c['total']:4d} fixed={c['fixed']:3d} broke={c['broke']:3d} "
              f"net_fixed={c['net_fixed']:+3d} consensus_to_wrong={c['consensus_to_wrong']:3d}")
    print("\n#### VOTING (mean±std) ####")
    for b in benches:
        v=master[b]["voting"]
        print(f"{b:14s} maj={v['majority']['mean']:.3f}±{v['majority']['std']:.3f}  "
              f"inf_based={v['influence_based']['mean']:.3f}±{v['influence_based']['std']:.3f}  "
              f"inf_aware={v['influence_aware']['mean']:.3f}±{v['influence_aware']['std']:.3f}")
    print(f"\n[OK] Findings -> {TAB_DIR}/analysis_findings.json")
    print(f"[OK] Figures  -> {FIG_DIR}/all_influence_vs_accuracy.png, pooled_influence_vs_accuracy.png, all_voting.png")


if __name__ == "__main__":
    main()
