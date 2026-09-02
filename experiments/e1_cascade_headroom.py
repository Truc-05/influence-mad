"""
E1 — Harmful-Cascade Headroom.

Claim (E1): debate net-destroys correct answers, and the *recoverable* accuracy
splits into two disjoint layers that motivate the journal:

    baseline (final majority)
      + VOTE-recoverable      : a correct answer still EXISTS at the final round
                                but majority vote misses it  -> a better AGGREGATOR
                                fixes it (this is conference territory).
      + ONLINE-recoverable    : a correct answer existed at round 0 but is EXTINCT
                                by the final round -> no vote can recover it; only
                                an ONLINE intervention during the debate can
                                (this is the journal's novelty).
      = online ceiling (a correct answer existed at some point in the debate).

Identity (exact, per question set):
    online_ceiling  = P(ind_any OR fin_any)
    posthoc_ceiling = P(fin_any)
    baseline        = P(fin_maj)
    vote_recoverable   = posthoc_ceiling - baseline = P(fin_any AND NOT fin_maj)
    online_recoverable = online_ceiling  - posthoc_ceiling = P(ind_any AND NOT fin_any)

This is a ZERO-GPU experiment: pure replay of existing debate logs. It also
reproduces the conference cascade numbers (broke / fixed / consensus_to_wrong)
as a correctness check against results/tables/analysis_findings.json.

Output:
    results/tables/e1_cascade_headroom.json
    results/figures/e1_headroom.png
"""
import json, glob, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io_utils import read_jsonl
from src.utils.scoring import is_correct

BENCH_ORDER = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa"]
FIG_DIR = "results/figures"; TAB_DIR = "results/tables"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)


def _norm(t):
    return (t or "").strip().lower()


def seed_logs(bench):
    """All per-seed logs pooled (each (seed, question) counts as one item),
    matching analyze_all.error_cascade so numbers are directly comparable."""
    out = []
    for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
        out.extend(read_jsonl(f))
    return out


def _maj(ans):
    """Majority answer among a dict {agent: answer}. Replicates
    analyze_all._maj exactly (modal normalized value, first original match)."""
    vals = [_norm(x) for x in ans.values()]
    key = max(set(vals), key=vals.count)
    for _, x in ans.items():
        if _norm(x) == key:
            return x
    return list(ans.values())[0]


def headroom(bench):
    logs = seed_logs(bench)
    n = 0
    # decomposition counters
    base = posthoc = online = round0_any = 0
    vote_rec = online_rec = 0
    # classic cascade (sanity vs conference)
    broke = fixed = revival = consensus_to_wrong = 0

    for lg in logs:
        gt = lg["ground_truth"]; ca = lg.get("correct_answers")
        ind = lg.get("independent_answers", {}); fin = lg.get("final_answers", {})
        if not ind or not fin:
            continue
        n += 1

        ind_any = any(is_correct(ind[a], gt, bench, ca) for a in ind)
        fin_any = any(is_correct(fin[a], gt, bench, ca) for a in fin)
        ind_maj = is_correct(_maj(ind), gt, bench, ca)
        fin_maj = is_correct(_maj(fin), gt, bench, ca)

        base += fin_maj
        posthoc += fin_any
        online += (ind_any or fin_any)
        round0_any += ind_any
        vote_rec += (fin_any and not fin_maj)
        online_rec += (ind_any and not fin_any)

        broke += (ind_maj and not fin_maj)
        fixed += ((not ind_maj) and fin_maj)
        revival += ((not ind_any) and fin_any)
        consensus_to_wrong += (ind_any and not fin_any)

    def rate(x):
        return x / n if n else 0.0

    return {
        "n": n,
        "counts": {
            "baseline_maj_correct": base,
            "posthoc_ceiling_any_correct": posthoc,
            "online_ceiling_ever_correct": online,
            "round0_any_correct": round0_any,
            "vote_recoverable": vote_rec,
            "online_recoverable": online_rec,
            "broke": broke, "fixed": fixed, "revival": revival,
            "consensus_to_wrong": consensus_to_wrong,
        },
        "rates": {
            "baseline": rate(base),
            "posthoc_ceiling": rate(posthoc),
            "online_ceiling": rate(online),
            "round0_any": rate(round0_any),
            "vote_recoverable": rate(vote_rec),
            "online_recoverable": rate(online_rec),
        },
    }


def main():
    benches = [b for b in BENCH_ORDER if glob.glob(f"data/debate_logs/{b}_seed*.jsonl")]
    res = {b: headroom(b) for b in benches}

    # pooled over all benchmarks (weight by #questions)
    tot = {k: sum(res[b]["counts"][k] for b in benches)
           for k in res[benches[0]]["counts"]}
    N = sum(res[b]["n"] for b in benches)
    pooled = {"n": N,
              "baseline": tot["baseline_maj_correct"] / N,
              "posthoc_ceiling": tot["posthoc_ceiling_any_correct"] / N,
              "online_ceiling": tot["online_ceiling_ever_correct"] / N,
              "vote_recoverable": tot["vote_recoverable"] / N,
              "online_recoverable": tot["online_recoverable"] / N,
              "counts": tot}

    out = {"benchmarks": res, "pooled": pooled}
    json.dump(out, open(f"{TAB_DIR}/e1_cascade_headroom.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)

    # ---------- figure: stacked headroom decomposition per benchmark ----------
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(benches)); w = 0.62
    base = [res[b]["rates"]["baseline"] for b in benches]
    vote = [res[b]["rates"]["vote_recoverable"] for b in benches]
    onl = [res[b]["rates"]["online_recoverable"] for b in benches]
    ax.bar(x, base, w, label="baseline (final majority)", color="#4C72B0")
    ax.bar(x, vote, w, bottom=base, label="vote-recoverable (better aggregator)", color="#DD8452")
    ax.bar(x, onl, w, bottom=np.add(base, vote),
           label="online-recoverable (needs intervention)", color="#C44E52")
    for i, b in enumerate(benches):
        ceil = base[i] + vote[i] + onl[i]
        ax.text(i, ceil + 0.01, f"{ceil:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(benches, rotation=12)
    ax.set_ylabel("Fraction of questions"); ax.set_ylim(0, 1)
    ax.set_title("E1 — Recovery headroom: baseline + vote-recoverable + online-recoverable")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/e1_headroom.png", dpi=150); plt.close(fig)

    # ---------------------------- print summary -------------------------------
    print("\n#### E1 HEADROOM (pooled over 5 seeds, per benchmark) ####")
    print(f"{'benchmark':14s} {'N':>4} {'base':>6} {'+vote':>7} {'+online':>8} "
          f"{'=ceil':>7} | {'broke':>5} {'fixed':>5} {'ext(c2w)':>8} {'revival':>7}")
    for b in benches:
        r = res[b]["rates"]; c = res[b]["counts"]; n = res[b]["n"]
        print(f"{b:14s} {n:>4} {r['baseline']:>6.1%} {r['vote_recoverable']:>7.1%} "
              f"{r['online_recoverable']:>8.1%} {r['online_ceiling']:>7.1%} | "
              f"{c['broke']:>5} {c['fixed']:>5} {c['consensus_to_wrong']:>8} {c['revival']:>7}")
    print("-" * 92)
    print(f"{'POOLED':14s} {pooled['n']:>4} {pooled['baseline']:>6.1%} "
          f"{pooled['vote_recoverable']:>7.1%} {pooled['online_recoverable']:>8.1%} "
          f"{pooled['online_ceiling']:>7.1%}")

    print("\n#### DECISION (E1 gate) ####")
    print(f"Baseline final-majority accuracy (pooled)      : {pooled['baseline']:.1%}")
    print(f"Post-hoc ceiling (best aggregator, same debate): {pooled['posthoc_ceiling']:.1%}  "
          f"(+{pooled['vote_recoverable']:.1%} vote-recoverable)")
    print(f"Online ceiling (preserve any-ever-correct)     : {pooled['online_ceiling']:.1%}  "
          f"(+{pooled['online_recoverable']:.1%} ONLINE-only, journal target)")
    verdict = "PASS" if pooled["online_recoverable"] >= 0.05 else "WEAK"
    print(f"\nOnline-only headroom = {pooled['online_recoverable']:.1%}  ->  gate: {verdict} "
          f"(threshold 5%)")
    print(f"\n[OK] table  -> {TAB_DIR}/e1_cascade_headroom.json")
    print(f"[OK] figure -> {FIG_DIR}/e1_headroom.png")


if __name__ == "__main__":
    main()
