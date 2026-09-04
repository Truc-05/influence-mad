"""
E31 — Compute-matched SELF-CONSISTENCY baseline (reviewer control). GPU via pinned Ollama :11436 (GPU 0).

Question (KBS reviewer): is multi-agent debate just an inefficient use of compute vs. sampling ONE model
N times independently (self-consistency, Wang et al. 2023)? We sample each debate backbone N=9 times
independently (temperature 0.7) on the SAME questions the debates used, majority-vote, and compare at
matched inference budget to: round-0 majority (3 calls) and the debated final majority (~8.7 calls).

Budgets reported: N in {3,5,9}. N=3 matches round-0 majority's 3 calls; N=9 matches the full debate.
Identical extraction/scoring as every other experiment (robust_answer + is_correct + bucket). Resumable
(per-sample cache). GPU: pinned :11436 = GPU 0 (idle); GPU 1's job untouched.

  CUDA_VISIBLE_DEVICES="" python experiments/e31_self_consistency.py
  (client-only; generation happens on the :11436 server which is GPU-0-pinned)
"""
import os, sys, glob, json, pickle
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.utils.io_utils import read_jsonl
from src.utils.answer_extract import robust_answer
from src.utils.scoring import is_correct
from src.agents.peer_agent import PeerAgent
from experiments.e4a_extinction_anatomy import bucket
import experiments.e3_router as e3

BASE_URL = "http://localhost:11436"          # GPU-0-pinned ollama (idle GPU); DO NOT use :11435 (GPU 1 busy)
BENCHES = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa"]
MODELS = [("careful", "qwen2.5:3b-instruct"), ("skeptic", "phi4-mini:latest"), ("bigpicture", "llama3.2:3b")]
N_SAMPLES = 9
CAP = 100                     # questions per benchmark (same qids as the frozen E_orig seed42 logs)
TEMP = 0.7                    # standard self-consistency sampling temperature
CACHE = "results/cache/e31_self_consistency.pkl"
OUT = "results/tables/e31_self_consistency.json"


def load_questions(bench):
    f = f"data/debate_logs/{bench}_seed42.jsonl"
    rows = []
    for lg in read_jsonl(f):
        rows.append({"qid": lg["question_id"], "q": lg["question"], "gt": lg["ground_truth"],
                     "ca": lg.get("correct_answers"), "lg": lg})
        if len(rows) >= CAP:
            break
    return rows


def debate_baselines(rows, bench):
    """Round-0 and final majority accuracy on this SAME subset, from the frozen logs."""
    r0_ok = fin_ok = 0
    for r in rows:
        p = e3.prep_log(r["lg"], bench); F, ag = p["F"], p["agents"]
        b0 = [p["B"][0][a] for a in ag if p["B"][0][a] is not None]
        bF = [p["B"][F][a] for a in ag if p["B"][F][a] is not None]
        if b0:
            w = Counter(b0).most_common(1)[0][0]
            r0_ok += 1 if any(p["C"][0][a] for a in ag if p["B"][0][a] == w) else 0
        if bF:
            w = Counter(bF).most_common(1)[0][0]
            fin_ok += 1 if any(p["C"][F][a] for a in ag if p["B"][F][a] == w) else 0
    n = len(rows)
    return {"round0_majority": round(r0_ok / n, 3), "final_majority": round(fin_ok / n, 3), "n": n}


def sc_accuracy(samples, gt, ca, bench, N):
    """samples: list of (bucket, correct) for one (model or mixed) at one question. Majority over first N."""
    use = samples[:N]
    buckets = [b for b, _ in use if b is not None]
    if not buckets:
        return 0
    win = Counter(buckets).most_common(1)[0][0]
    return 1 if any(c for b, c in use if b == win) else 0


def run():
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    agents = {p: PeerAgent(model_name=m, role=p, temperature=TEMP, base_url=BASE_URL,
                           max_tokens=512, timeout=180, persona=p) for p, m in MODELS}
    n_new = 0
    per_bench = {}
    bench_rows = {b: load_questions(b) for b in BENCHES}
    # GENERATION PHASE — loop model-outermost so each model stays resident (all 3 fit in 16 GB; no swap thrash)
    for persona, _ in MODELS:
        for bench in BENCHES:
            for r in bench_rows[bench]:
                for s in range(N_SAMPLES):
                    key = (bench, r["qid"], persona, s)
                    if key in cache:
                        continue
                    try:
                        resp = agents[persona].solve(r["q"])
                        ra = robust_answer(resp.answer, resp.reasoning, bench)
                        bk = bucket(ra, bench)
                        ok = (ra is not None) and bool(is_correct(ra, r["gt"], bench, r["ca"]))
                        cache[key] = (bk, ok)
                    except Exception:
                        cache[key] = (None, False)
                    n_new += 1
                    if n_new % 200 == 0:
                        pickle.dump(cache, open(CACHE, "wb"))
                        print(f"  ... {n_new} new gens (model={persona} bench={bench})", flush=True)
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"  [done model={persona}] total new gens so far {n_new}", flush=True)
    # EVALUATION PHASE
    for bench in BENCHES:
        rows = bench_rows[bench]
        base = debate_baselines(rows, bench)
        res = {"baselines": base, "sc": {}}
        for persona, _ in MODELS:
            for N in (3, 5, 9):
                acc = sum(sc_accuracy([cache[(bench, r["qid"], persona, s)] for s in range(N_SAMPLES)],
                                      r["gt"], r["ca"], bench, N) for r in rows) / len(rows)
                res["sc"].setdefault(persona, {})[f"N={N}"] = round(acc, 3)
        # mixed self-consistency: 3 samples from each of the 3 models = 9-call heterogeneous ensemble
        for N in (3, 6, 9):
            per = N // 3
            def mixed(r):
                sm = []
                for persona, _ in MODELS:
                    sm += [cache[(bench, r["qid"], persona, s)] for s in range(per)]
                return sc_accuracy(sm, r["gt"], r["ca"], bench, len(sm))
            res["sc"].setdefault("mixed_3models", {})[f"N={N}"] = round(sum(mixed(r) for r in rows) / len(rows), 3)
        per_bench[bench] = res
        print(f"[{bench}] round0={base['round0_majority']} final={base['final_majority']}  "
              f"SC(qwen N=9)={res['sc']['careful']['N=9']} SC(best N=9)="
              f"{max(res['sc'][p]['N=9'] for p,_ in MODELS)} mixed N=9={res['sc']['mixed_3models']['N=9']}", flush=True)

    # pooled
    pooled = {"baselines": {}, "sc": defaultdict(dict)}
    ns = {b: per_bench[b]["baselines"]["n"] for b in BENCHES}
    tot = sum(ns.values())
    for k in ("round0_majority", "final_majority"):
        pooled["baselines"][k] = round(sum(per_bench[b]["baselines"][k] * ns[b] for b in BENCHES) / tot, 3)
    for who in list(per_bench[BENCHES[0]]["sc"].keys()):
        for Nk in per_bench[BENCHES[0]]["sc"][who]:
            pooled["sc"][who][Nk] = round(sum(per_bench[b]["sc"][who][Nk] * ns[b] for b in BENCHES) / tot, 3)
    result = {"config": {"n_samples": N_SAMPLES, "cap_per_bench": CAP, "temp": TEMP, "budgets": [3, 5, 9],
                         "models": dict(MODELS)}, "per_bench": per_bench, "pooled": pooled}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print("\n==== POOLED ====")
    print("round0_majority", pooled["baselines"]["round0_majority"], " final_majority", pooled["baselines"]["final_majority"])
    for who, d in pooled["sc"].items():
        print(f"  SC {who:14}", d)
    print(f"[OK] {OUT}")


def paired_ci():
    """Zero-GPU, cache-only: paired bootstrap CI + McNemar for SC (N=9) vs debated final majority,
    pooled over the same 500-question subset. Clustering unit = question (single seed, unique qids).
    Merges a 'paired_vs_debate' block into the existing OUT json; no server calls, no generation."""
    import numpy as np
    cache = pickle.load(open(CACHE, "rb"))
    recs = []
    for bench in BENCHES:
        for r in load_questions(bench):
            p = e3.prep_log(r["lg"], bench); F, ag = p["F"], p["agents"]
            bF = [p["B"][F][a] for a in ag if p["B"][F][a] is not None]
            deb = 0
            if bF:
                w = Counter(bF).most_common(1)[0][0]
                deb = 1 if any(p["C"][F][a] for a in ag if p["B"][F][a] == w) else 0
            rec = {"deb": deb}
            for persona, _ in MODELS:
                sm9 = [cache[(bench, r["qid"], persona, s)] for s in range(N_SAMPLES)]
                rec[persona] = sc_accuracy(sm9, r["gt"], r["ca"], bench, 9)
            mixed = []
            for persona, _ in MODELS:
                mixed += [cache[(bench, r["qid"], persona, s)] for s in range(3)]
            rec["mixed"] = sc_accuracy(mixed, r["gt"], r["ca"], bench, len(mixed))
            recs.append(rec)
    n = len(recs)
    deb = np.array([x["deb"] for x in recs])
    rng = np.random.default_rng(20260902)
    idxb = rng.integers(0, n, size=(4000, n))
    from scipy.stats import binomtest
    out = {"n": n, "clustering": "question (single seed)", "B": 4000}
    for who in ("mixed", "bigpicture", "careful", "skeptic"):
        a = np.array([x[who] for x in recs]); d = a - deb
        stats = d[idxb].mean(axis=1)
        win = int(((a == 1) & (deb == 0)).sum()); loss = int(((a == 0) & (deb == 1)).sum())
        pval = binomtest(min(win, loss), win + loss, 0.5).pvalue if (win + loss) else 1.0
        out[f"{who}_N9_vs_debate"] = {"delta": round(float(d.mean()), 3),
                                      "ci": [round(float(np.percentile(stats, 2.5)), 3),
                                             round(float(np.percentile(stats, 97.5)), 3)],
                                      "mcnemar_win": win, "mcnemar_loss": loss, "mcnemar_p": float(pval)}
    res = json.load(open(OUT)); res["paired_vs_debate"] = out
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[paired CI, n={n}]  debate={deb.mean():.3f}")
    for k, v in out.items():
        if k.endswith("_vs_debate"):
            print(f"  {k:22} Δ={v['delta']:+.3f} {v['ci']}  McNemar {v['mcnemar_win']}/{v['mcnemar_loss']} p={v['mcnemar_p']:.2e}")
    print(f"[OK] merged into {OUT}")


if __name__ == "__main__":
    if "--ci" in sys.argv:
        paired_ci()
    else:
        run()
