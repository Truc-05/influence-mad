"""E25 breadth-extension VERDICT (zero-GPU) for AQuA/MMLU/ARC.

Reuses the FROZEN e18/e22/e23 machinery (method frozen at E23) with EXPLICIT per-cell logdirs, and
applies the PRE-REGISTERED disposition rule (paper/E25_newbench_prereg.md). Writes a SEPARATE table
results/tables/e25_newbench.json — it NEVER touches the frozen 15-cell results/tables/e25_generalization.json.

  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e25_newbench_verdict.py
"""
import os, sys, glob, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.io_utils import read_jsonl
from src.utils.answer_extract import robust_answer
import experiments.e18_hypothesis_preservation as e18
import experiments.e22_provenance as e22
import experiments.e23_provenance_selector as e23

OUT = "results/tables/e25_newbench.json"
BENCHES = ["aqua", "mmlu", "arc"]
FROZEN = "data/debate_logs"
GEN = "data/debate_logs/gen"

# explicit (label -> per-bench logdir). E_orig: aqua reuses frozen seed42; mmlu/arc = generated 3B.
def logdir_for(label, bench):
    if label == "E_orig":
        return FROZEN if bench in ("aqua",) else f"{GEN}/E_orig3b"
    return f"{GEN}/{label}"

LABELS = ["E_orig", "E_7b", "E_mixed"]

# pre-registered validity-gate thresholds (paper/E25_newbench_prereg.md §3 Step A)
MAX_CENSORED = 0.15
MAJ_LO, MAJ_HI = 0.10, 0.95
AVAIL_FLOOR = 0.15          # CR_history minimum (extinction phenomenon testable)


def _censored_counts(bench, logdir):
    """(#censored, #total) FINAL-round agent answers that robust_answer cannot recover."""
    files = glob.glob(f"{logdir}/{bench}_seed*.jsonl")
    tot = cen = 0
    for fp in files:
        for lg in read_jsonl(fp):
            final = lg["rounds"][-1]["answers"]
            for a, o in final.items():
                tot += 1
                if robust_answer(o["answer"], o["reasoning"], bench) is None:
                    cen += 1
    return cen, tot


def cell_verdict(label, bench):
    logdir = logdir_for(label, bench)
    if not glob.glob(f"{logdir}/{bench}_seed*.jsonl"):
        return None
    cen, tot = _censored_counts(bench, logdir)
    cell = {"label": label, "bench": bench, "logdir": logdir,
            "censored_rate": round(cen / tot, 3) if tot else 1.0, "_cen": [cen, tot]}
    rows = e18.rows_for(bench, logdir=logdir)
    if rows:
        s = e18.summarize(rows, f"H1 {label}/{bench}")
        cell["H1"] = {"n": s["n"], "CR_final": s["CR_final"], "CR_history": s["CR_history"],
                      "dCR": s["dCR"], "dCR_ci": s["dCR_ci"], "maj_acc": s["maj_acc_mode"],
                      "pass": bool(s["dCR_ci"][0] > 0)}
    crows = e22.candidates_for(bench, logdir=logdir)
    if crows:
        s = e22.summarize(crows, f"H2 {label}/{bench}")
        ni = s["signals"]["n_independent"]
        cell["H2"] = {"n_candidates": s["n_candidates"], "base_rate": s["base_rate"],
                      "n_independent": {"dir_auc": ni["dir_auc"], "dir_ci": ni["dir_ci"],
                                        "monotone": ni["monotone_pooled"]},
                      "combo_auc": s["untuned_combo_auc"],
                      "pass": bool(ni["dir_ci"][0] > 0.5)}
    deb = e23.build_debates(benches=[bench], logdir=logdir)
    if deb:
        prov = e23.add_provenance_picks(deb)
        s = e23.summarize(deb, f"H3 {label}/{bench}", ["final_majority", "round0_majority"] + prov)
        eq, lg = s["prov_equalweight"], s["prov_logistic_xfit"]
        cell["H3"] = {"n": s["n"], "final_majority": s["final_majority"]["acc"], "CR_history": s["CR_history"],
                      "prov_equalweight": {"acc": eq["acc"], "delta": eq["delta"], "ci": eq["ci"]},
                      "prov_logistic_xfit": {"acc": lg["acc"], "delta": lg["delta"], "ci": lg["ci"]},
                      "pass": bool(eq["ci"][0] > 0 or lg["ci"][0] > 0)}
    return cell


def _hold(cells, key):
    """H_k 'holds' for a bench iff it passes in >=2 of the available ensemble cells."""
    have = [c for c in cells if key in c]
    p = sum(1 for c in have if c[key]["pass"])
    return p, len(have)


def disposition(bench, cells):
    """Apply the pre-registered rule; returns (verdict_str, reason, metrics)."""
    if not cells:
        return "NO_DATA", "no logs yet", {}
    # POOLED validity metrics across ensemble cells (prereg §3 Step A): censored = answer-count
    # pooled; maj_acc / CR_history = n(debate)-weighted pooled.
    tc = sum(c["_cen"][0] for c in cells); tt = sum(c["_cen"][1] for c in cells)
    cen = (tc / tt) if tt else 1.0
    h1 = [c["H1"] for c in cells if "H1" in c]
    wn = sum(x["n"] for x in h1) or 1
    maj = sum(x["maj_acc"] * x["n"] for x in h1) / wn if h1 else 0.0
    crh = sum(x["CR_history"] * x["n"] for x in h1) / wn if h1 else 0.0
    m = {"pooled_censored": round(cen, 3), "pooled_maj_acc": round(maj, 3), "pooled_CR_history": round(crh, 3)}
    # Step A: validity gate
    if cen > MAX_CENSORED:
        return "EXCLUDED", f"parse-fail: censored {cen:.2f} > {MAX_CENSORED}", m
    if not (MAJ_LO <= maj <= MAJ_HI):
        return "EXCLUDED", f"degenerate acc: maj {maj:.2f} outside [{MAJ_LO},{MAJ_HI}]", m
    if crh < AVAIL_FLOOR:
        return "EXCLUDED", f"availability floor: CR_history {crh:.2f} < {AVAIL_FLOOR}", m
    # Step B: classify by hypothesis signs (>=2/3 ensembles)
    h1p, h1n = _hold(cells, "H1"); h2p, h2n = _hold(cells, "H2"); h3p, h3n = _hold(cells, "H3")
    h1 = h1p >= 2 and h1p >= (h1n - h1p + 1)
    h2 = h2p >= 2 and h2p >= (h2n - h2p + 1)
    h3 = h3p >= 2 and h3p >= (h3n - h3p + 1)
    m["holds"] = {"H1": f"{h1p}/{h1n}", "H2": f"{h2p}/{h2n}", "H3": f"{h3p}/{h3n}"}
    if h1 and h2 and h3:
        return "MAIN", "H1&H2&H3 all hold", m
    if h1 and h2 and not h3:
        return "S9_BOUNDARY", "H1&H2 hold, H3 fails (applicability boundary)", m
    return "APPENDIX", f"characterized negative (H1={h1},H2={h2},H3={h3})", m


def run():
    out = {"benches": BENCHES, "labels": LABELS, "cells": {}, "disposition": {},
           "prereg": "paper/E25_newbench_prereg.md"}
    for bench in BENCHES:
        cells = []
        for label in LABELS:
            c = cell_verdict(label, bench)
            if c is None:
                print(f"[--  ] {label}/{bench}: no logs yet, skip")
                continue
            cells.append(c)
            out["cells"].setdefault(bench, {})[label] = c
        verdict, reason, metrics = disposition(bench, cells)
        out["disposition"][bench] = {"verdict": verdict, "reason": reason, "metrics": metrics}

    print("\n" + "=" * 72 + "\nE25 NEW-BENCH — pre-registered disposition\n" + "=" * 72)
    for bench in BENCHES:
        d = out["disposition"][bench]
        print(f"  {bench:6} -> {d['verdict']:12} | {d['reason']}")
        if d["metrics"]:
            print(f"           {d['metrics']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")
    return out


if __name__ == "__main__":
    run()
