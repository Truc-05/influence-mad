"""
E25 — GENERALIZATION of the FROZEN HP-MAD / provenance method (E18/E22/E23).

The method is FROZEN at E23. E25 does NOT tune or invent anything. It re-generates debates for
ALTERNATIVE model-family ensembles (and, later, other N-agent counts / benchmarks) and re-runs the
EXACT frozen verdict code to test three pre-committed hypotheses, cell by cell:

  H1  extinction generalizes         : CR_history > CR_final, i.e. ΔCR > 0        (reuse e18)
  H2  provenance -> correctness       : n_independent dirAUC > chance & monotone   (reuse e22)
  H3  provenance selector still helps : prov selector Acc > final-majority Acc      (reuse e23)

The point is the SIGN of each hypothesis across cells, not a winning configuration. A cell = one
(ensemble, benchmark, seed). Alt-ensemble logs are written to a TAGGED path so the frozen originals
in data/debate_logs/*.jsonl are NEVER touched:

    data/debate_logs/gen/<ensemble>/<bench>_seed<seed>.jsonl

E_orig is the reference cell and REUSES the existing frozen logs (data/debate_logs) — no regen.

GPU DISCIPLINE ([[gpu-run-pinning]]): generation goes ONLY to the gpu1-pinned Ollama at
http://localhost:11435 (never :11434 / gpu0). Verdict is ZERO-GPU.

Usage:
  # 0) wiring self-test (mock backend, no GPU): gen tiny logs + run the full verdict on them
  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e25_generalization.py --selftest

  # 1) generate one cell (gpu1-pinned). Resumable: re-run to continue where it stopped.
  /home/alex/venvs/env/bin/python experiments/e25_generalization.py \
        --gen --ensemble E_7b --bench gsm8k --seed 42 --n 100

  # 2) generate the whole pilot (loops the PILOT matrix; skips E_orig = reuses frozen logs)
  /home/alex/venvs/env/bin/python experiments/e25_generalization.py --gen-pilot

  # 3) verdict over whatever cells exist (zero-GPU), writes results/tables/e25_generalization.json
  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e25_generalization.py --verdict
"""
import os, sys, json, argparse, random
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from src.utils.io_utils import read_jsonl
from src.utils.seed import set_seed
import src.data_collection.run_debate as rd
import experiments.e18_hypothesis_preservation as e18
import experiments.e22_provenance as e22
import experiments.e23_provenance_selector as e23

# --------------------------------------------------------------------------- config
GPU1_URL = "http://localhost:11435"          # gpu1-pinned server ONLY (never :11434)
ORIG_LOGDIR = "data/debate_logs"             # frozen E_orig logs (reused, never written)
GEN_ROOT = "data/debate_logs/gen"            # tagged alt-ensemble logs
OUT = "results/tables/e25_generalization.json"
MAX_ROUNDS = 3
MAX_TOKENS = 512
TIMEOUT = 180

# Ensembles keep the 3 frozen PERSONAS (careful/skeptic/bigpicture) and swap the BACKBONE model,
# breaking the persona-perpendicular-backbone confound of the frozen run. All models are already
# pulled on :11435. (mistral-small3.1:24b deliberately excluded — barely fits 16GB, very slow.)
ENSEMBLES = {
    # reference cell = the frozen ensemble; verdict reuses existing logs, NOT regenerated.
    "E_orig":  [("careful", "qwen2.5:3b-instruct", 0.3), ("skeptic", "phi4-mini:latest", 0.5),
                ("bigpicture", "llama3.2:3b", 0.5)],
    # ~7-8B tier, three different families.
    "E_7b":    [("careful", "qwen2.5:7b", 0.3), ("skeptic", "mistral:7b", 0.5),
                ("bigpicture", "llama3.1:8b", 0.5)],
    # mixed size/family: one 9B + two small.
    "E_mixed": [("careful", "gemma2:9b", 0.3), ("skeptic", "phi4-mini:latest", 0.5),
                ("bigpicture", "llama3.2:3b", 0.5)],
    # (expansion, only if the pilot PASSes)
    "E_14b":   [("careful", "qwen2.5:14b", 0.3), ("skeptic", "gemma2:9b", 0.5),
                ("bigpicture", "llama3.1:8b", 0.5)],
    "E_homog": [("careful", "qwen2.5:7b", 0.3), ("skeptic", "qwen2.5:7b", 0.5),
                ("bigpicture", "qwen2.5:7b", 0.5)],
    # strong-lite: one 14B anchor + two small. Fits 16GB (9.0+2.5+2.0=13.5GB). Deliberately
    # parallels E_mixed (Gemma9B->Qwen14B, same Phi+Llama3B) to isolate the strong-anchor effect.
    "E_strong": [("careful", "qwen2.5:14b", 0.3), ("skeptic", "phi4-mini:latest", 0.5),
                 ("bigpicture", "llama3.2:3b", 0.5)],
    # N=5 structural stress test: E_orig's 3 + two mid agents. 4-tuple = (name, model, temp,
    # persona); names UNIQUE (run_debate keys by name) but personas reused (PERSONAS has only 3).
    # Fits 16GB (1.9+2.5+2.0+4.4+4.7=15.5GB).
    "E_n5": [("careful", "qwen2.5:3b-instruct", 0.3, "careful"),
             ("skeptic", "phi4-mini:latest", 0.5, "skeptic"),
             ("bigpicture", "llama3.2:3b", 0.5, "bigpicture"),
             ("careful2", "mistral:7b", 0.4, "careful"),
             ("skeptic2", "qwen2.5:7b", 0.3, "skeptic")],
}

PILOT_ENSEMBLES = ["E_orig", "E_7b", "E_mixed"]
PILOT_BENCHES = ["gsm8k", "strategyqa", "commonsenseqa"]
PILOT_SEED = 42
PILOT_N = 100


def _agents_spec(ensemble):
    # spec = (name, model, temp) or (name, model, temp, persona). name must be UNIQUE (run_debate
    # keys agents by name); persona defaults to name (valid for the 3 canonical personas).
    specs = []
    for spec in ENSEMBLES[ensemble]:
        name, m, t = spec[0], spec[1], spec[2]
        persona = spec[3] if len(spec) > 3 else name
        specs.append({"name": name, "model": m, "persona": persona, "temperature": t})
    return specs


def _cell_config(ensemble, backend, base_url):
    return {"agents": _agents_spec(ensemble),
            "debate": {"backend": backend, "base_url": base_url, "max_rounds": MAX_ROUNDS,
                       "max_tokens": MAX_TOKENS, "timeout_seconds": TIMEOUT}}


def logdir_for(ensemble):
    return ORIG_LOGDIR if ensemble == "E_orig" else f"{GEN_ROOT}/{ensemble}"


# --------------------------------------------------------------------------- generation (GPU)
def gen_cell(ensemble, bench, seed, n, backend="ollama", base_url=GPU1_URL):
    """Generate (or resume) one (ensemble, bench, seed) cell into the TAGGED path. gpu1-pinned.
    Reuses the frozen debate primitives so logs are byte-compatible with e3.prep_log."""
    if ensemble == "E_orig":
        print(f"[skip] E_orig/{bench} reuses frozen logs in {ORIG_LOGDIR} (never regenerated)")
        return None
    set_seed(seed)
    outdir = logdir_for(ensemble)
    os.makedirs(outdir, exist_ok=True)
    output_path = f"{outdir}/{bench}_seed{seed}.jsonl"
    data = read_jsonl(f"data/processed/{bench}.jsonl")[:n]
    logs, done = rd._load_existing(output_path)
    remaining = [it for it in data if it["question_id"] not in done]
    if not remaining:
        print(f"[done] {ensemble}/{bench}_seed{seed}: complete ({len(logs)}/{len(data)})")
        return output_path
    print(f"[gen ] {ensemble}/{bench}_seed{seed}: {len(done)} done, {len(remaining)} remaining "
          f"(backend={backend}, url={base_url})")
    config = _cell_config(ensemble, backend, base_url)
    agents = rd._build_agents(config) if backend != "mock" else None
    with open(output_path, "a", encoding="utf-8") as f:
        for i, it in enumerate(remaining):
            try:
                if backend == "mock":
                    rng = random.Random(f"{seed}-{it['question_id']}")
                    log = rd._mock_single_debate(it["question_id"], it["question"], it["ground_truth"],
                                                 config["agents"], MAX_ROUNDS, rng,
                                                 correct_answers=it.get("correct_answers"))
                else:
                    log = rd._run_single_debate(it["question_id"], it["question"], it["ground_truth"],
                                                agents, MAX_ROUNDS,
                                                correct_answers=it.get("correct_answers"))
            except Exception as e:
                print(f"  [{i+1}/{len(remaining)}] {it['question_id']} FAILED: {e}")
                continue
            f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n"); f.flush()
    print(f"[ok  ] {ensemble}/{bench}_seed{seed}: {output_path}")
    return output_path


def gen_pilot(backend="ollama", base_url=GPU1_URL, n=PILOT_N, seed=PILOT_SEED):
    for ens in PILOT_ENSEMBLES:
        for bench in PILOT_BENCHES:
            gen_cell(ens, bench, seed, n, backend=backend, base_url=base_url)


# --------------------------------------------------------------------------- verdict (zero-GPU)
def _cell_verdict(ensemble, bench):
    """Run the three FROZEN verdicts on one cell's logs (via the logdir= seam). Returns None if no logs."""
    logdir = logdir_for(ensemble)
    import glob as _g
    if not _g.glob(f"{logdir}/{bench}_seed*.jsonl"):
        return None
    cell = {"ensemble": ensemble, "bench": bench, "logdir": logdir}

    # H1 — extinction / preservation ceiling (e18)
    rows = e18.rows_for(bench, logdir=logdir)
    if rows:
        s = e18.summarize(rows, f"H1 {ensemble}/{bench}")
        cell["H1"] = {"n": s["n"], "CR_final": s["CR_final"], "CR_history": s["CR_history"],
                      "dCR": s["dCR"], "dCR_ci": s["dCR_ci"], "maj_acc": s["maj_acc_mode"],
                      "pass": bool(s["dCR_ci"][0] > 0)}

    # H2 — provenance predicts correctness (e22); primary signal = n_independent
    crows = e22.candidates_for(bench, logdir=logdir)
    if crows:
        s = e22.summarize(crows, f"H2 {ensemble}/{bench}")
        ni = s["signals"]["n_independent"]; npr = s["signals"]["n_persist"]
        cell["H2"] = {"n_candidates": s["n_candidates"], "base_rate": s["base_rate"],
                      "n_independent": {"dir_auc": ni["dir_auc"], "dir_ci": ni["dir_ci"],
                                        "monotone": ni["monotone_pooled"]},
                      "n_persist": {"dir_auc": npr["dir_auc"], "dir_ci": npr["dir_ci"],
                                    "monotone": npr["monotone_pooled"]},
                      "combo_auc": s["untuned_combo_auc"],
                      "pass": bool(ni["dir_ci"][0] > 0.5)}

    # H3 — provenance selector still beats final-majority (e23)
    deb = e23.build_debates(benches=[bench], logdir=logdir)
    if deb:
        prov = e23.add_provenance_picks(deb)
        s = e23.summarize(deb, f"H3 {ensemble}/{bench}", ["final_majority", "round0_majority"] + prov)
        eq, lg = s["prov_equalweight"], s["prov_logistic_xfit"]
        cell["H3"] = {"n": s["n"], "final_majority": s["final_majority"]["acc"], "CR_history": s["CR_history"],
                      "prov_equalweight": {"acc": eq["acc"], "delta": eq["delta"], "ci": eq["ci"]},
                      "prov_logistic_xfit": {"acc": lg["acc"], "delta": lg["delta"], "ci": lg["ci"]},
                      "round0_majority": {"acc": s["round0_majority"]["acc"],
                                          "delta": s["round0_majority"]["delta"]},
                      "pass": bool(eq["ci"][0] > 0 or lg["ci"][0] > 0)}
    return cell


def verdict(ensembles=None, benches=None):
    ensembles = ensembles or PILOT_ENSEMBLES
    benches = benches or PILOT_BENCHES
    cells, out = [], {"cells": {}, "ensembles": ensembles, "benches": benches}
    for ens in ensembles:
        for bench in benches:
            c = _cell_verdict(ens, bench)
            if c is None:
                print(f"[--  ] {ens}/{bench}: no logs yet, skip")
                continue
            cells.append(c)
            out["cells"].setdefault(ens, {})[bench] = c

    # ---- cross-cell SIGN summary: does each hypothesis hold in most cells? ----
    def _tally(key):
        done = [c for c in cells if key in c]
        passed = [c for c in done if c[key]["pass"]]
        return len(passed), len(done), [f"{c['ensemble']}/{c['bench']}" for c in done if not c[key]["pass"]]

    summary = {}
    for h, desc in [("H1", "extinction generalizes (ΔCR>0)"),
                    ("H2", "provenance->correctness (n_indep dirAUC-lo>0.5)"),
                    ("H3", "prov selector beats final-majority (ΔAcc CI-lo>0)")]:
        p, d, fails = _tally(h)
        summary[h] = {"desc": desc, "pass_cells": p, "total_cells": d, "failing_cells": fails}
    out["summary"] = summary

    print("\n" + "=" * 72 + "\nE25 GENERALIZATION — cross-cell sign summary\n" + "=" * 72)
    for h in ["H1", "H2", "H3"]:
        s = summary[h]
        print(f"  {h}: {s['pass_cells']}/{s['total_cells']} cells hold  — {s['desc']}"
              + (f"   (fails: {', '.join(s['failing_cells'])})" if s["failing_cells"] else ""))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")
    return out


# --------------------------------------------------------------------------- self-test (no GPU)
def selftest():
    """Mock-generate a tiny alt-ensemble cell and run the full verdict on it. Validates the
    generation path + the logdir= seam + all three reused verdicts end-to-end, ZERO GPU."""
    print("=== E25 SELFTEST (mock backend, no GPU) ===")
    tag = "E_selftest"
    ENSEMBLES[tag] = ENSEMBLES["E_orig"]           # any spec; mock ignores the model name
    d = f"{GEN_ROOT}/{tag}"
    for bench in ["gsm8k", "strategyqa"]:
        # clean any stale selftest file so the mock regen is deterministic
        p = f"{d}/{bench}_seed42.jsonl"
        if os.path.exists(p):
            os.remove(p)
        gen_cell(tag, bench, 42, 40, backend="mock")
    v = verdict(ensembles=[tag], benches=["gsm8k", "strategyqa"])
    ok = True
    for bench in ["gsm8k", "strategyqa"]:
        c = v["cells"].get(tag, {}).get(bench)
        assert c and "H1" in c and "H2" in c and "H3" in c, f"missing verdict for {bench}"
        assert -1.0 <= c["H1"]["dCR"] <= 1.0
        assert 0.0 <= c["H3"]["final_majority"] <= 1.0
    print("\n[SELFTEST PASS] gen + logdir seam + e18/e22/e23 reuse all wired correctly.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gen", action="store_true", help="generate one cell")
    ap.add_argument("--gen-pilot", action="store_true", help="generate the whole pilot matrix")
    ap.add_argument("--verdict", action="store_true", help="run frozen verdicts over existing cells")
    ap.add_argument("--ensemble", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--seed", type=int, default=PILOT_SEED)
    ap.add_argument("--n", type=int, default=PILOT_N)
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--base-url", default=GPU1_URL)
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if args.gen:
        assert args.ensemble and args.bench, "--gen needs --ensemble and --bench"
        gen_cell(args.ensemble, args.bench, args.seed, args.n, backend=args.backend, base_url=args.base_url)
        return
    if args.gen_pilot:
        gen_pilot(backend=args.backend, base_url=args.base_url, n=args.n, seed=args.seed); return
    if args.verdict:
        verdict(); return
    ap.print_help()


if __name__ == "__main__":
    main()
