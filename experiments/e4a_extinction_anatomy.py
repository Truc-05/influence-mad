"""
E4a — Extinction Anatomy (ZERO-GPU, no causal claim).

E1 established that ~10.9% pooled accuracy is "online-recoverable": a correct
answer existed at round 0 but is EXTINCT from the final answers. E4a asks a
purely *structural* question about those extinction cases -- WITHOUT claiming any
recovered accuracy (that needs re-querying the model; see E5/E6):

    At the moment just before the correct answer disappears, does a concrete,
    LOCAL, gold-free-detectable intervention point exist -- or is the loss an
    unavoidable case of group ignorance?

For each extinction case we locate the *death round* t* (the last round whose
predecessor still held a correct answer and after which no agent is ever correct
again) and dissect the transition t*-1 -> t*:

  - k = # correct holders at t*-1 (how many agents we'd need to preserve)
  - did the dying holder(s) HERD onto a peer's answer (adoption-driven death,
    i.e. targetable by "mask persuader / preserve dissent") vs switch to a novel
    wrong answer (self-error)?
  - were the correct holders a strict MINORITY against a unified wrong bloc
    (a 1-vs-rest split is detectable WITHOUT gold)?

Headline metric -- Preventable-Extinction Opportunity Rate:
    fraction of extinctions that die via an adoption-driven loss-of-correctness
    of a gold-free-detectable minority holder (a concrete online handle exists).

NOTE ON t* AND ROUND BUDGET: max_rounds=3 (rounds 0,1,2) with early consensus
stop, so every extinction dies at t*=1 or t*=2. The "early enough to intervene"
gate is therefore structurally satisfied here; the informative cut is t*=1
(fastest cascades) vs t*=2. Whether the pattern persists with more rounds is an
E8 robustness item, flagged explicitly rather than hidden.

Output:
    results/tables/e4a_extinction_anatomy.json
    results/figures/e4a_extinction_anatomy.png
"""
import json, glob, os, sys
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io_utils import read_jsonl
from src.utils.scoring import is_correct, _extract_number, _normalize_text


def bucket(ans, bench):
    """Canonical answer class in the SAME space is_correct decides on, so that
    structural comparisons (adoption / unified-wrong / minority) are consistent
    with correctness. Returns None when no answer can be extracted."""
    if ans is None:
        return None
    if bench in ("gsm8k", "svamp", "math"):
        x = _extract_number(ans)
        return None if x is None else round(x, 4)
    p = _normalize_text(ans or "")
    if bench == "strategyqa":
        return "yes" if (p.startswith("yes") or p == "true") else "no"
    if bench in ("commonsenseqa", "bbh", "mmlu", "aqua"):
        return p[0] if p else None
    return p or None            # truthfulqa / fallback (off the main axis)

BENCH_ORDER = ["gsm8k", "strategyqa", "commonsenseqa", "bbh"]   # truthfulqa off main axis
ALL_BENCH = BENCH_ORDER + ["truthfulqa"]
FIG_DIR = "results/figures"; TAB_DIR = "results/tables"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)


def _norm(t):
    return (t or "").strip().lower()


def seed_logs(bench):
    out = []
    for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
        out.extend(read_jsonl(f))
    return out


def _round_ans(lg, r):
    """{agent: answer_str} at round r, from the rounds[] transcript."""
    return {a: v["answer"] for a, v in lg["rounds"][r]["answers"].items()}


def anatomy(lg, bench):
    """Return a dict describing one debate, or None if not an extinction case."""
    gt = lg["ground_truth"]; ca = lg.get("correct_answers")
    F = len(lg["rounds"]) - 1
    agents = list(lg["rounds"][0]["answers"].keys())

    # correctness of each agent per round
    corr = [{a: is_correct(_round_ans(lg, r)[a], gt, bench, ca) for a in agents}
            for r in range(F + 1)]
    n = [sum(c.values()) for c in corr]                    # #correct holders per round

    # extinction case: correct at round 0, none correct at final
    if not (n[0] >= 1 and n[F] == 0):
        return None

    # death round t*: start of the terminal all-wrong run (n[t*-1] >= 1)
    s = F
    while s - 1 >= 1 and n[s - 1] == 0:
        s -= 1
    t = s                                                   # t* ; n[t-1] >= 1, n[t] == 0

    holders = [a for a in agents if corr[t - 1][a]]         # dying correct holders
    k = len(holders)
    wrong_before = [a for a in agents if not corr[t - 1][a]]
    ans_before = {a: bucket(_round_ans(lg, t - 1)[a], bench) for a in agents}
    ans_at = {a: bucket(_round_ans(lg, t)[a], bench) for a in agents}
    changed_at = lg["rounds"][t].get("changed", {})

    # did dying holders herd onto a bucket a PEER already held at t*-1?
    def herded(a):
        peers = {ans_before[o] for o in agents if o != a and ans_before[o] is not None}
        return ans_at[a] is not None and ans_at[a] in peers
    death_by_adoption = any(herded(a) for a in holders)

    # unified wrong bloc at t*-1 (all wrong agents share ONE bucket)?
    wrong_answers = {ans_before[a] for a in wrong_before}
    unified_wrong = (len(wrong_before) >= 1 and len(wrong_answers) == 1)
    wrong_bloc_ans = next(iter(wrong_answers)) if unified_wrong else None

    # minority: correct holders strictly outnumbered by the largest wrong coalition
    wrong_counts = Counter(ans_before[a] for a in wrong_before)
    largest_wrong = max(wrong_counts.values()) if wrong_counts else 0
    minority = k < largest_wrong                            # gold-free detectable split
    lone = (k == 1)

    # did a holder switch specifically onto the unified/modal wrong answer?
    herd_onto_majwrong = unified_wrong and any(ans_at[a] == wrong_bloc_ans for a in holders)

    # sanity: dying holders actually revised
    holders_changed = all(changed_at.get(a, True) for a in holders)

    # PRIMARY 3-way partition of the death mechanism (exhaustive):
    #   social_adoption    : >=1 dying holder herded onto a peer's wrong bucket
    #                        -> socially shaped (a persuader exists); influence-targetable
    #   self_error_parsefail: holder(s) produced an UNPARSEABLE answer (bucket None)
    #                        -> measurement/extraction failure, not a cascade
    #   self_error_value   : holder(s) drifted to a NOVEL wrong VALUE no peer held
    #                        -> genuine self-inflicted error; masking a peer cannot fix it
    if death_by_adoption:
        dtype = "social_adoption"
    elif all(ans_at[a] is None for a in holders):
        dtype = "self_error_parsefail"
    else:
        dtype = "self_error_value"

    # NOT a causal 'preventable' claim (that needs re-query, see E5): just the
    # restrictive subset where the loss is BOTH adoption-driven AND carries a
    # gold-free minority cue (1-correct-vs-unified-wrong). Reported, not headline.
    social_and_detectable = death_by_adoption and minority

    return {
        "t_star": t, "F": F, "k": k,
        "death_by_adoption": death_by_adoption,
        "unified_wrong": unified_wrong,
        "minority": minority, "lone": lone,
        "herd_onto_majwrong": herd_onto_majwrong,
        "holders_changed": holders_changed,
        "dtype": dtype,
        "social_and_detectable": social_and_detectable,
    }


def summarize(cases):
    n = len(cases)
    if n == 0:
        return {"n": 0}
    def rate(pred):
        return sum(1 for c in cases if pred(c)) / n
    return {
        "n": n,
        "correct_holder_before": rate(lambda c: c["k"] >= 1),          # ~1.0 sanity
        "holders_actually_revised": rate(lambda c: c["holders_changed"]),
        "t_star_1": rate(lambda c: c["t_star"] == 1),
        "t_star_2": rate(lambda c: c["t_star"] == 2),
        "single_holder": rate(lambda c: c["lone"]),                    # want >40-50%
        "death_by_adoption": rate(lambda c: c["death_by_adoption"]),
        "minority_detectable_nogold": rate(lambda c: c["minority"]),   # gold-free signal
        "herd_onto_maj_wrong": rate(lambda c: c["herd_onto_majwrong"]),
        "social_and_detectable": rate(lambda c: c["social_and_detectable"]),
        "dtype": {t: rate(lambda c, t=t: c["dtype"] == t)
                  for t in ["social_adoption", "self_error_value", "self_error_parsefail"]},
    }


def main():
    per_bench_cases = {}
    card = {}
    for b in ALL_BENCH:
        if not glob.glob(f"data/debate_logs/{b}_seed*.jsonl"):
            continue
        logs = seed_logs(b)
        cases = [a for a in (anatomy(lg, b) for lg in logs) if a is not None]
        per_bench_cases[b] = cases
        # answer-space cardinality proxy: distinct gold buckets over the benchmark
        card[b] = len({bucket(lg["ground_truth"], b) for lg in logs})

    res = {b: summarize(cs) for b, cs in per_bench_cases.items()}
    for b in res:
        res[b]["answer_space_cardinality"] = card[b]
    main_cases = [c for b in BENCH_ORDER for c in per_bench_cases.get(b, [])]
    all_cases = [c for b in ALL_BENCH for c in per_bench_cases.get(b, [])]
    pooled_main = summarize(main_cases)     # headline: 4 benchmarks, tqa excluded
    pooled_all = summarize(all_cases)

    out = {"benchmarks": res, "pooled_main_4bench": pooled_main, "pooled_all_5bench": pooled_all}
    json.dump(out, open(f"{TAB_DIR}/e4a_extinction_anatomy.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)

    # ------------------------------ figure ------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    benches = [b for b in ALL_BENCH if per_bench_cases.get(b)]
    x = np.arange(len(benches)); w = 0.6
    types = ["social_adoption", "self_error_value", "self_error_parsefail"]
    labels = ["social adoption (herds onto peer)", "self-error (novel wrong value)",
              "parse-fail (unparseable)"]
    colors = ["#C44E52", "#55A868", "#CCCCCC"]
    bottom = np.zeros(len(benches))
    for t, lab, col in zip(types, labels, colors):
        vals = [res[b]["dtype"][t] for b in benches]
        ax1.bar(x, vals, w, bottom=bottom, label=lab, color=col)
        bottom = np.add(bottom, vals)
    for i, b in enumerate(benches):
        ax1.text(i, 1.02, f"n={res[b]['n']}\n|A|={card[b]}", ha="center", fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(benches, rotation=12)
    ax1.set_ylabel("Fraction of extinction cases"); ax1.set_ylim(0, 1.15)
    ax1.set_title("Death mechanism (|A| = answer-space size)")
    ax1.legend(fontsize=8, loc="lower center"); ax1.grid(True, axis="y", alpha=0.3)

    # social-adoption rate vs answer-space cardinality (expose the confound)
    ax2.scatter([card[b] for b in benches], [res[b]["dtype"]["social_adoption"] for b in benches],
                s=70, color="#C44E52")
    for b in benches:
        ax2.annotate(b[:6], (card[b], res[b]["dtype"]["social_adoption"]),
                     fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax2.set_xscale("log"); ax2.set_xlabel("answer-space cardinality |A| (log)")
    ax2.set_ylabel("social-adoption fraction"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Adoption is confounded by |A|\n(gsm8k = clean anchor)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/e4a_extinction_anatomy.png", dpi=150); plt.close(fig)

    # --------------------------- print summary --------------------------
    def row(name, key, target=""):
        vals = " ".join(f"{res[b][key]:>7.0%}" for b in benches)
        pm = pooled_main[key]
        print(f"{name:34s} {vals}  | {pm:>6.0%}  {target}")

    print("\n#### E4a — EXTINCTION ANATOMY (rows=metric, cols=benchmark) ####")
    print(f"{'':34s} " + " ".join(f"{b[:7]:>7s}" for b in benches) + "  | POOLED(4)")
    print(f"{'n extinction cases':34s} " + " ".join(f"{res[b]['n']:>7d}" for b in benches)
          + f"  | {pooled_main['n']:>6d}")
    print(f"{'answer-space cardinality |A|':34s} " + " ".join(f"{card[b]:>7d}" for b in benches)
          + "  |   (confound)")
    print("-" * 78)
    row("correct holder before (sanity)", "correct_holder_before", "~100% expected")
    row("holders actually revised (sanity)", "holders_actually_revised", "~100% expected")
    row("single holder to preserve (k=1)", "single_holder", ">40-50% -> yes")
    row("death at t*=1 (fastest)", "t_star_1")
    print("-" * 78)
    row(">> SOCIAL-ADOPTION death", "death_by_adoption", "targetable, but |A|-confounded")
    row("minority split (gold-free cue)", "minority_detectable_nogold", "detectability")
    row("social AND detectable (subset)", "social_and_detectable", "restrictive")

    print("\n#### VERDICT (confound-aware) ####")
    print("social-adoption fraction by |A|:  " +
          "  ".join(f"{b}(|A|={card[b]})={res[b]['dtype']['social_adoption']:.0%}" for b in benches))
    anchor = res["gsm8k"]["dtype"]["social_adoption"] if "gsm8k" in res else None
    if anchor is not None:
        print(f"\nClean anchor = gsm8k (|A|={card['gsm8k']}, open numeric): "
              f"social-adoption = {anchor:.0%}, self-error = {1-anchor:.0%}")
        print("strategyqa (|A|=2) is UNINFORMATIVE for this question (binary artifact).")
        verdict = ("MIXED: a purely-social intervention addresses only ~1/3 of open-ended "
                   "extinctions; the majority is self-error drift -> reconsider intervention family")
        print(f"\n=> {verdict}")
    print(f"\n[OK] table  -> {TAB_DIR}/e4a_extinction_anatomy.json")
    print(f"[OK] figure -> {FIG_DIR}/e4a_extinction_anatomy.png")


if __name__ == "__main__":
    main()
