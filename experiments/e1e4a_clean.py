"""
Clean recompute of E1 (headroom) + E4a (mechanism) under ROBUST answer
extraction, with truncation/parse failures CENSORED out of behavioral headroom.

Why: E4a found ~18% of gsm8k "extinctions" were the death-round holder emitting
an unparseable "REASONING:" (generation truncated before the ANSWER: line) -- a
capture artifact, not a behavioral loss of the correct hypothesis. Counting them
inflates the online-recoverable headroom. Here we:

  1. re-score every round with robust_answer() (recovers \\boxed{} etc.; flags the
     rest as censored),
  2. recompute E1 headroom,
  3. split extinction into  {behavioral: social_adoption | self_error_value}  vs
     {censored}, and define BEHAVIORAL online-recoverable = extinction minus
     censored -- the number to freeze for the journal,
  4. print naive-vs-clean deltas.

Output: results/tables/e1e4a_clean.json, results/figures/e1e4a_clean.png
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
from src.utils.answer_extract import robust_answer
from experiments.e4a_extinction_anatomy import bucket

BENCH_ORDER = ["gsm8k", "strategyqa", "commonsenseqa", "bbh"]
ALL_BENCH = BENCH_ORDER + ["truthfulqa"]
FIG_DIR = "results/figures"; TAB_DIR = "results/tables"


def seed_logs(bench):
    out = []
    for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl")):
        out.extend(read_jsonl(f))
    return out


def _ansobj(lg, r, a):
    return lg["rounds"][r]["answers"][a]


def analyze(bench, robust):
    """Return per-question booleans + (for robust) extinction mechanism tags."""
    logs = seed_logs(bench)
    base = posthoc = online = onl_rec = vote_rec = n = 0
    ext_social = ext_self = ext_censored = 0

    for lg in logs:
        gt = lg["ground_truth"]; ca = lg.get("correct_answers")
        F = len(lg["rounds"]) - 1
        agents = list(lg["rounds"][0]["answers"].keys())

        def ans(r, a):
            o = _ansobj(lg, r, a)
            return robust_answer(o["answer"], o["reasoning"], bench) if robust else o["answer"]

        def ok(r, a):
            s = ans(r, a)
            return (s is not None) and is_correct(s, gt, bench, ca)

        corr = [{a: ok(r, a) for a in agents} for r in range(F + 1)]
        ncorr = [sum(c.values()) for c in corr]
        ind_any, fin_any = ncorr[0] >= 1, ncorr[F] >= 1

        # majority (bucket-based, deterministic first-occurrence tie-break)
        def maj_ok(r):
            bks = [bucket(ans(r, a), bench) for a in agents]
            bks = [b for b in bks if b is not None]
            if not bks:
                return False
            key = max(bks, key=bks.count)
            # score an original answer whose bucket == key
            for a in agents:
                if bucket(ans(r, a), bench) == key:
                    return ok(r, a)
            return False
        fin_maj = maj_ok(F)

        n += 1
        base += fin_maj
        posthoc += fin_any
        online += (ind_any or fin_any)
        vote_rec += (fin_any and not fin_maj)
        onl_rec += (ind_any and not fin_any)

        # mechanism of extinction (robust only)
        if robust and ind_any and not fin_any:
            s = F
            while s - 1 >= 1 and ncorr[s - 1] == 0:
                s -= 1
            t = s
            holders = [a for a in agents if corr[t - 1][a]]
            bt = {a: bucket(ans(t, a), bench) for a in agents}         # None => censored
            bp = {a: bucket(ans(t - 1, a), bench) for a in agents}
            if all(bt[a] is None for a in holders):
                ext_censored += 1
            elif any(bt[a] is not None and bt[a] in {bp[o] for o in agents if o != a and bp[o] is not None}
                     for a in holders):
                ext_social += 1
            else:
                ext_self += 1

    r = {"n": n, "baseline": base / n, "posthoc": posthoc / n, "online": online / n,
         "vote_rec": vote_rec / n, "onl_rec": onl_rec / n,
         "counts": {"base": base, "posthoc": posthoc, "online": online,
                    "vote_rec": vote_rec, "onl_rec": onl_rec, "n": n}}
    if robust:
        ext = ext_social + ext_self + ext_censored
        r["ext"] = {"social": ext_social, "self": ext_self, "censored": ext_censored, "total": ext}
        r["behavioral_onl_rec"] = (ext_social + ext_self) / n     # censored removed
        r["counts"]["behavioral_onl_rec"] = ext_social + ext_self
    return r


def main():
    benches = [b for b in ALL_BENCH if glob.glob(f"data/debate_logs/{b}_seed*.jsonl")]
    naive = {b: analyze(b, robust=False) for b in benches}
    clean = {b: analyze(b, robust=True) for b in benches}

    def pooled(d, keys):
        tot = {k: sum(d[b]["counts"][k] for b in BENCH_ORDER) for k in keys}
        N = sum(d[b]["counts"]["n"] for b in BENCH_ORDER)
        return tot, N
    _, N = pooled(naive, ["n"])
    n_onl = sum(naive[b]["counts"]["onl_rec"] for b in BENCH_ORDER) / N
    c_onl = sum(clean[b]["counts"]["onl_rec"] for b in BENCH_ORDER) / N
    c_beh = sum(clean[b]["counts"]["behavioral_onl_rec"] for b in BENCH_ORDER) / N
    c_soc = sum(clean[b]["ext"]["social"] for b in BENCH_ORDER)
    c_slf = sum(clean[b]["ext"]["self"] for b in BENCH_ORDER)
    c_cen = sum(clean[b]["ext"]["censored"] for b in BENCH_ORDER)
    c_base_naive = sum(naive[b]["counts"]["base"] for b in BENCH_ORDER) / N
    c_base_clean = sum(clean[b]["counts"]["base"] for b in BENCH_ORDER) / N

    out = {"naive": naive, "clean": clean,
           "pooled4": {"N": N, "onl_rec_naive": n_onl, "onl_rec_clean": c_onl,
                       "behavioral_onl_rec": c_beh,
                       "baseline_naive": c_base_naive, "baseline_clean": c_base_clean,
                       "ext_social": c_soc, "ext_self": c_slf, "ext_censored": c_cen}}
    os.makedirs(TAB_DIR, exist_ok=True)
    json.dump(out, open(f"{TAB_DIR}/e1e4a_clean.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)

    # ---------------- print ----------------
    print("\n#### CLEAN RECOMPUTE — online-recoverable headroom (naive vs robust) ####")
    print(f"{'bench':14s} {'naive':>7} {'clean':>7} {'censored':>9} {'BEHAVIORAL':>11} | "
          f"{'social':>7} {'self':>6}")
    for b in benches:
        c = clean[b]; ext = c["ext"]
        cen = ext["censored"] / c["ext"]["total"] if ext["total"] else 0
        soc = ext["social"] / c["ext"]["total"] if ext["total"] else 0
        slf = ext["self"] / c["ext"]["total"] if ext["total"] else 0
        tag = "  <- off-axis" if b == "truthfulqa" else ""
        print(f"{b:14s} {naive[b]['onl_rec']:>7.1%} {c['onl_rec']:>7.1%} {cen:>9.0%} "
              f"{c['behavioral_onl_rec']:>11.1%} | {soc:>7.0%} {slf:>6.0%}{tag}")
    print("-" * 70)
    print(f"{'POOLED(4)':14s} {n_onl:>7.1%} {c_onl:>7.1%} "
          f"{c_cen/(c_soc+c_slf+c_cen):>9.0%} {c_beh:>11.1%} | "
          f"{c_soc/(c_soc+c_slf):>7.0%} {c_slf/(c_soc+c_slf):>6.0%}  (of behavioral)")

    print("\n#### FROZEN JOURNAL NUMBERS (pooled 4-bench, robust) ####")
    print(f"baseline majority : naive {c_base_naive:.1%} -> clean {c_base_clean:.1%}")
    print(f"online-recoverable: naive {n_onl:.1%} -> clean {c_onl:.1%} "
          f"-> BEHAVIORAL {c_beh:.1%}  (censored parser artifact = {c_onl-c_beh:.1%})")
    print(f"behavioral extinction mechanism: social-adoption {c_soc}  self-error {c_slf}  "
          f"(+ censored {c_cen})")
    print(f"gsm8k clean anchor: social {clean['gsm8k']['ext']['social']}, "
          f"self {clean['gsm8k']['ext']['self']}, censored {clean['gsm8k']['ext']['censored']} "
          f"of {clean['gsm8k']['ext']['total']}")

    # ---------------- figure: extinction decomposition as fraction of questions ----------------
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(benches)); w = 0.6
    soc = [clean[b]["ext"]["social"] / clean[b]["counts"]["n"] for b in benches]
    slf = [clean[b]["ext"]["self"] / clean[b]["counts"]["n"] for b in benches]
    cen = [clean[b]["ext"]["censored"] / clean[b]["counts"]["n"] for b in benches]
    ax.bar(x, soc, w, label="behavioral · social adoption", color="#C44E52")
    ax.bar(x, slf, w, bottom=soc, label="behavioral · self-error", color="#55A868")
    ax.bar(x, cen, w, bottom=np.add(soc, slf), label="censored (parser artifact)", color="#CCCCCC")
    for i, b in enumerate(benches):
        ax.text(i, soc[i] + slf[i] + cen[i] + 0.002,
                f"{(soc[i]+slf[i]):.1%}\nbehav.", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(benches, rotation=12)
    ax.set_ylabel("Fraction of ALL questions")
    ax.set_title("Extinction headroom, cleaned: behavioral (recoverable target) vs censored")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/e1e4a_clean.png", dpi=150); plt.close(fig)
    print(f"\n[OK] {TAB_DIR}/e1e4a_clean.json ; {FIG_DIR}/e1e4a_clean.png")


if __name__ == "__main__":
    main()
