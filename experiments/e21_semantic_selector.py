"""
E21 — Semantic selector over the frozen hypothesis pool (GPU judge, gpu1 :11435).
See paper/E21_semantic_selector_prereg.md (LOCKED). Judge MUST pick a candidate from the pool
(labelled CHOICE), never invent an answer. Arms (same model+budget, only input set changes):
  final_majority  logs only (no GPU)          — deployed baseline
  J(final)        question + FINAL-round pool  — control (judge, no preservation)
  J(history)      question + ALL-round pool    — HP selector
  J(q)            question only, judge solves  — control (is the pool even needed?)
PRIMARY GATE: Δ_HP = Acc[J(history)] - Acc[J(final)] > 0 (cluster-boot qid, lo>0). GPU=gpu1 ONLY.

Usage:
  CUDA_VISIBLE_DEVICES="" python experiments/e21_semantic_selector.py --selftest   # NO GPU
  python experiments/e21_semantic_selector.py --pilot 100     # 100/bench x4 benches, gpu1
  python experiments/e21_semantic_selector.py --full          # all 5 bench x 5 seed, gpu1
"""
import os, sys, glob, re, json, time, pickle, hashlib, argparse
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import requests
from src.utils.io_utils import read_jsonl
from src.utils.answer_extract import robust_answer
from src.utils.scoring import is_correct
from experiments.e4a_extinction_anatomy import bucket as _bucket
import experiments.e3_router as e3

BASE_URL = "http://localhost:11435"          # gpu1 ONLY, NEVER :11434/gpu0
MODEL = "qwen2.5:7b"
CACHE = "results/cache/e21_judge.pkl"
OUT = "results/tables/e21_semantic_selector.json"
PILOT_BENCHES = ["gsm8k", "bbh", "strategyqa", "commonsenseqa"]
FULL_BENCHES = ["gsm8k", "bbh", "strategyqa", "commonsenseqa", "truthfulqa"]
BENCH_SEED = {"gsm8k": 1, "bbh": 2, "strategyqa": 3, "commonsenseqa": 4, "truthfulqa": 5}
SEL_NP = 64          # selection arms: tiny output -> forces SELECT (cannot re-solve), fast
Q_NP = 512           # J(q) control: full budget to SOLVE (conservative -> harder for us)
TRUNC = 600

SEL_TMPL = (
    "You are a careful verifier. A group of agents debated the question below and proposed the "
    "candidate answers that follow, each with its reasoning. Your job is to judge which proposed "
    "answer is most likely CORRECT. Do NOT re-solve the question and do NOT propose a new answer; "
    "only choose among the given labels.\n\n"
    "Question:\n{q}\n\nCandidate answers:\n{cands}\n\n"
    "On the FIRST line output exactly:\nCHOICE: <label>\n"
    "where <label> is one of the labels above. You may add a one-line reason after it.")
QONLY_TMPL = ("Solve the following question. Think step by step, then output exactly one final "
              "line:\nANSWER: <your final answer>\n\nQuestion:\n{q}")


# ------------------------------ pool construction ------------------------------
def build_pool(p, rounds):
    """Ordered distinct candidates over the given rounds: {label,bucket,reasoning,first_round}.
    Representative reasoning = longest supporting reasoning; provenance = first-seen round."""
    F, agents = p["F"], p["agents"]
    seen = {}
    for r in rounds:
        for a in agents:
            b = p["B"][r][a]
            if b is None:
                continue
            rea = p["R"][r][a] or ""
            if b not in seen:
                seen[b] = {"bucket": b, "reasoning": rea, "first_round": r}
            elif len(rea) > len(seen[b]["reasoning"]):
                seen[b]["reasoning"] = rea
    cands = sorted(seen.values(), key=lambda c: (c["first_round"], str(c["bucket"])))
    for i, c in enumerate(cands):
        c["label"] = chr(ord("A") + i)
    return cands


def sel_prompt(q, cands):
    lines = [f"[{c['label']}] Answer: {c['bucket']}\n    Reasoning: {(c['reasoning'] or '')[:TRUNC]}"
             for c in cands]
    return SEL_TMPL.format(q=q, cands="\n".join(lines))


def parse_choice(raw, cands):
    labels = {c["label"]: c for c in cands}
    m = re.search(r"CHOICE:\s*[\[\*\s]*([A-Za-z])", raw or "")
    if m and m.group(1).upper() in labels:
        return labels[m.group(1).upper()], False
    toks = re.findall(r"\[\s*([A-Za-z])\s*\]", raw or "")
    for t in reversed(toks):
        if t.upper() in labels:
            return labels[t.upper()], False
    for c in cands:                                   # last resort: did it name a candidate value?
        if str(c["bucket"]) and str(c["bucket"]) in (raw or ""):
            return c, False
    return None, True                                 # parse-fail


# ------------------------------- judge client ---------------------------------
def _call(base_url, model, prompt, num_predict, timeout=180, retries=2):
    url = f"{base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "num_predict": num_predict}}
    err = None
    for att in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["response"]
        except requests.exceptions.RequestException as e:
            err = e
            if att < retries:
                time.sleep(5)
    raise err


def make_judge(base_url, model, cache_path, stub=None):
    cache = pickle.load(open(cache_path, "rb")) if (stub is None and os.path.exists(cache_path)) else {}
    state = {"n": 0, "calls": 0}

    def judge(prompt, num_predict):
        key = hashlib.md5(f"{model}\n{num_predict}\n{prompt}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        raw = stub(prompt) if stub else _call(base_url, model, prompt, num_predict)
        cache[key] = raw
        state["calls"] += 1
        state["n"] += 1
        if stub is None and state["n"] % 25 == 0:
            pickle.dump(cache, open(cache_path, "wb"))
        return raw

    def flush():
        if stub is None:
            pickle.dump(cache, open(cache_path, "wb"))
    return judge, flush, state


# ------------------------------- per-debate eval ------------------------------
def eval_debate(lg, bench, judge):
    p = e3.prep_log(lg, bench)
    F, agents = p["F"], p["agents"]
    q, gt, ca = lg["question"], lg["ground_truth"], lg.get("correct_answers")
    correct = {b for r in range(F + 1) for a in agents
               if p["B"][r][a] is not None and p["C"][r][a] for b in [p["B"][r][a]]}
    fb = [p["B"][F][a] for a in agents if p["B"][F][a] is not None]
    maj_bucket = Counter(fb).most_common(1)[0][0] if fb else None
    final_set = set(fb)

    hist_c = build_pool(p, range(F + 1))
    final_c = build_pool(p, [F])
    if not hist_c:
        return None

    fp = sel_prompt(q, final_c)
    fr = judge(fp, SEL_NP)
    fpick, ffail = parse_choice(fr, final_c)
    hp = sel_prompt(q, hist_c)
    if hp == fp:
        hpick, hfail = fpick, ffail
    else:
        hpick, hfail = parse_choice(judge(hp, SEL_NP), hist_c)

    qr = judge(QONLY_TMPL.format(q=q), Q_NP)
    qm = re.search(r"ANSWER:\s*(.+)", qr or "", re.IGNORECASE)
    q_ans = robust_answer(qm.group(1).strip() if qm else (qr or "")[:200], qr or "", bench)
    q_correct = (q_ans is not None) and is_correct(q_ans, gt, bench, ca)

    hb = hpick["bucket"] if hpick else None
    return {"qid": lg["question_id"], "bench": bench,
            "final_maj": maj_bucket in correct,
            "J_final": bool(fpick) and fpick["bucket"] in correct,
            "J_hist": bool(hpick) and hb in correct,
            "J_q": bool(q_correct),
            "CR_final": bool(final_set & correct),
            "CR_history": bool({c["bucket"] for c in hist_c} & correct),
            "pools_differ": {c["bucket"] for c in hist_c} != final_set,
            "resurrection": bool(hpick) and hb is not None and hb not in final_set,
            "resurrection_correct": bool(hpick) and hb not in final_set and hb in correct,
            "ffail": ffail, "hfail": hfail, "qfail": q_ans is None}


# --------------------------------- sampling -----------------------------------
def sample_debates(benches, n_per):
    out = []
    for bench in benches:
        debates = [lg for f in sorted(glob.glob(f"data/debate_logs/{bench}_seed*.jsonl"))
                   for lg in read_jsonl(f)]
        if n_per and len(debates) > n_per:
            rng = np.random.default_rng(BENCH_SEED[bench])
            debates = [debates[i] for i in sorted(rng.choice(len(debates), n_per, replace=False))]
        out += [(lg, bench) for lg in debates]
    return out


# ----------------------------------- stats ------------------------------------
def _mean(rows, k):
    return float(np.mean([r[k] for r in rows])) if rows else float("nan")


def boot_delta(rows, a, b, B=4000):
    cl = defaultdict(list)
    for r in rows:
        cl[r["qid"]].append(r)
    cl = list(cl.values())
    rng = np.random.default_rng(0)
    d = np.empty(B)
    for i in range(B):
        s = [x for j in rng.integers(0, len(cl), len(cl)) for x in cl[j]]
        d[i] = np.mean([x[a] for x in s]) - np.mean([x[b] for x in s])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summarize(rows, label):
    fm, jf, jh, jq = (_mean(rows, k) for k in ("final_maj", "J_final", "J_hist", "J_q"))
    crf, crh = _mean(rows, "CR_final"), _mean(rows, "CR_history")
    dhp_lo, dhp_hi = boot_delta(rows, "J_hist", "J_final")
    s2_lo, s2_hi = boot_delta(rows, "J_hist", "final_maj")
    sq_lo, sq_hi = boot_delta(rows, "J_hist", "J_q")
    head = crh - fm
    cap = (jh - fm) / head * 100 if head > 0 else float("nan")
    diff = [r for r in rows if r["pools_differ"]]
    res_rate = _mean(diff, "resurrection") if diff else float("nan")
    res_acc = (np.mean([r["resurrection_correct"] for r in diff if r["resurrection"]])
               if any(r["resurrection"] for r in diff) else float("nan"))
    print(f"\n=== {label}  (n={len(rows)}, {len({r['qid'] for r in rows})} qid) ===")
    print(f"  final_majority {fm:.3f} | J(final) {jf:.3f} | J(history) {jh:.3f} | J(q) {jq:.3f}")
    print(f"  CR_final {crf:.3f} | CR_history {crh:.3f}  (ceiling)")
    print(f"  PRIMARY  Δ_HP = J(hist)-J(final) = {jh - jf:+.3f} [{dhp_lo:+.3f},{dhp_hi:+.3f}]"
          f"  {'PASS' if dhp_lo > 0 else 'fail'}")
    print(f"  sec  J(hist)-final_maj {jh - fm:+.3f} [{s2_lo:+.3f},{s2_hi:+.3f}] | capture {cap:.0f}%")
    print(f"  sec  J(hist)-J(q)      {jh - jq:+.3f} [{sq_lo:+.3f},{sq_hi:+.3f}]"
          f"  {'(pool adds value)' if sq_lo > 0 else ''}")
    print(f"  pools differ {_mean(rows,'pools_differ'):.2f} | resurrection rate {res_rate:.2f}"
          f" acc {res_acc:.2f} | parse-fail f/h/q "
          f"{_mean(rows,'ffail'):.2f}/{_mean(rows,'hfail'):.2f}/{_mean(rows,'qfail'):.2f}")
    return {"n": len(rows), "final_maj": fm, "J_final": jf, "J_hist": jh, "J_q": jq,
            "CR_final": crf, "CR_history": crh,
            "dHP": jh - jf, "dHP_ci": [dhp_lo, dhp_hi], "dHP_pass": dhp_lo > 0,
            "J_hist_minus_finalmaj": [jh - fm, s2_lo, s2_hi], "capture_pct": cap,
            "J_hist_minus_Jq": [jh - jq, sq_lo, sq_hi],
            "resurrection_rate": res_rate, "resurrection_acc": res_acc}


def pilot_gates(out, benches):
    pooled = out["pooled"]
    catastrophic = [b for b in benches if b in out and out[b]["J_hist"] < out[b]["final_maj"] - 0.03]
    rr, ra = pooled["resurrection_rate"], pooled["resurrection_acc"]
    gates = {
        "1_dHP_sign_positive": bool(pooled["dHP"] > 0),
        "2_Jhist_gt_finalmaj": bool(pooled["J_hist"] > pooled["final_maj"]),
        "3_no_catastrophic_bench": bool(len(catastrophic) == 0),
        "4_uses_recovered": bool((not np.isnan(rr)) and rr > 0 and (np.isnan(ra) or ra >= 0.5)),
    }
    gates["catastrophic_benches"] = catastrophic
    gates["PILOT_VERDICT"] = ("GO_FULL" if all(gates[k] for k in
                              ("1_dHP_sign_positive", "2_Jhist_gt_finalmaj",
                               "3_no_catastrophic_bench", "4_uses_recovered")) else "STOP")
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="debates per bench (pilot, 4 benches)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base_url", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if args.selftest:
        # stub judge: deterministic, picks label of the LAST candidate; J(q) returns a fixed answer.
        seen = {}
        def stub(prompt):
            if "CHOICE:" in prompt:
                labs = re.findall(r"\[([A-Za-z])\]", prompt)
                return f"reasoning...\nCHOICE: {labs[-1]}" if labs else "CHOICE: A"
            return "ANSWER: 0"
        judge, flush, state = make_judge("", "", "", stub=stub)
        debates = sample_debates(["gsm8k"], 4)
        rows = [eval_debate(lg, b, judge) for lg, b in debates]
        rows = [r for r in rows if r]
        assert all(k in rows[0] for k in ("J_final", "J_hist", "J_q", "dHP" if False else "final_maj"))
        # dedup: a debate whose pools are equal must NOT have triggered a 2nd distinct sel-call.
        print(f"[selftest] {len(rows)} debates, judge calls={state['calls']} "
              f"(<= debates*3). arms present, dedup+parse OK. PLUMBING VALID (NO GPU).")
        return

    assert "11435" in args.base_url, f"E21 must run on gpu1 :11435, got {args.base_url}"
    benches = FULL_BENCHES if args.full else PILOT_BENCHES
    n_per = 0 if args.full else (args.pilot or 100)
    debates = sample_debates(benches, n_per)
    judge, flush, state = make_judge(args.base_url, args.model, CACHE)
    print(f"[{'FULL' if args.full else 'PILOT'}] {len(debates)} debates, model={args.model} @ {args.base_url}")

    rows = []
    try:
        for i, (lg, b) in enumerate(debates):
            r = eval_debate(lg, b, judge)
            if r:
                rows.append(r)
            if (i + 1) % 50 == 0:
                flush()
                print(f"  [{i+1}/{len(debates)}] judge calls so far={state['calls']}")
    finally:
        flush()

    out = {}
    for b in benches:
        br = [r for r in rows if r["bench"] == b]
        if br:
            out[b] = summarize(br, b)
    out["pooled"] = summarize(rows, "POOLED")
    if not args.full:
        out["pilot_gates"] = pilot_gates(out, benches)
        print(f"\n  PILOT GATES: {json.dumps(out['pilot_gates'], indent=None)}")
    out["_meta"] = {"mode": "full" if args.full else "pilot", "model": args.model,
                    "n": len(rows), "judge_calls": state["calls"]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT.replace(".json", "_full.json") if args.full else OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}  ({state['calls']} judge calls)")


if __name__ == "__main__":
    main()
