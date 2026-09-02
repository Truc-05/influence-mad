"""E25 model-scale + N-scaling VERDICT (zero-GPU) for E_strong (14B anchor) and E_n5 (N=5).

Reuses the FROZEN e18/e22/e23 machinery (method frozen at E23) with EXPLICIT per-cell logdirs, and the
SAME per-cell gates + Step-A validity gate as the pre-registered E25 breadth verdict
(paper/E25_newbench_prereg.md ADDENDUM). Writes a SEPARATE table results/tables/e25_scale.json — it
NEVER touches results/tables/e25_generalization.json or results/tables/e25_newbench.json.

Unlike the 4-new-bench verdict (a per-benchmark MAIN/BOUNDARY/EXCLUDED disposition across 3 ensembles),
E_strong/E_n5 are ROBUSTNESS AXES: the question is "does the phenomenon survive with a 14B anchor / with
N=5?". So the primary output is a PER-CELL sign summary across the full 9-bench set (parallels the Stage-1
15-cell summary), with the Step-A validity gate applied per cell.

  CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python experiments/e25_scale_verdict.py
"""
import os, sys, glob, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import experiments.e18_hypothesis_preservation as e18
import experiments.e22_provenance as e22
import experiments.e23_provenance_selector as e23
# reuse the pre-registered gate thresholds + censored counter from the breadth verdict (no drift)
from experiments.e25_newbench_verdict import (
    _censored_counts, MAX_CENSORED, MAJ_LO, MAJ_HI, AVAIL_FLOOR,
)

OUT = "results/tables/e25_scale.json"
GEN = "data/debate_logs/gen"
LABELS = ["E_strong", "E_n5"]
BENCHES = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa", "aqua", "arc", "mmlu"]


def cell_verdict(label, bench):
    """Per-cell H1/H2/H3 (identical gates to e25_newbench_verdict.cell_verdict) + Step-A validity."""
    logdir = f"{GEN}/{label}"
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

    # Step-A validity gate (per cell, prereg §3): does the pipeline/regime hold here at all?
    h1 = cell.get("H1", {})
    reasons = []
    if cell["censored_rate"] > MAX_CENSORED:
        reasons.append(f"censored {cell['censored_rate']:.2f}>{MAX_CENSORED}")
    if h1 and not (MAJ_LO <= h1["maj_acc"] <= MAJ_HI):
        reasons.append(f"maj {h1['maj_acc']:.2f} outside[{MAJ_LO},{MAJ_HI}]")
    if h1 and h1["CR_history"] < AVAIL_FLOOR:
        reasons.append(f"CR_hist {h1['CR_history']:.2f}<{AVAIL_FLOOR}")
    cell["valid"] = not reasons
    cell["invalid_reason"] = "; ".join(reasons)
    return cell


def _flag(cell, key):
    if key not in cell:
        return "  - "
    return "  ✓ " if cell[key]["pass"] else "  ✗ "


def run():
    out = {"labels": LABELS, "benches": BENCHES, "cells": {}, "summary": {},
           "prereg": "paper/E25_newbench_prereg.md (ADDENDUM)"}
    for label in LABELS:
        for bench in BENCHES:
            c = cell_verdict(label, bench)
            if c is None:
                print(f"[--  ] {label}/{bench}: no logs, skip")
                continue
            out["cells"].setdefault(label, {})[bench] = c

    # ---- per-cell table ----
    print("\n" + "=" * 78)
    print("E25 SCALE (E_strong 14B-anchor, E_n5 N=5) — per-cell verdict, full 9-bench set")
    print("=" * 78)
    print(f"{'ensemble/bench':24} {'valid':6} {'H1':4} {'H2':4} {'H3':4}  notes")
    for label in LABELS:
        for bench in BENCHES:
            c = out["cells"].get(label, {}).get(bench)
            if not c:
                continue
            v = "OK" if c["valid"] else "INVAL"
            note = "" if c["valid"] else c["invalid_reason"]
            print(f"{label + '/' + bench:24} {v:6} {_flag(c,'H1')} {_flag(c,'H2')} {_flag(c,'H3')} {note}")

    # ---- sign summary (per ensemble + pooled) ----
    print("\n" + "-" * 78)
    print("SIGN SUMMARY  (H_k pass count; H3 counted among Step-A-VALID cells)")
    print("-" * 78)
    pooled = {"H1": [0, 0], "H2": [0, 0], "H3": [0, 0], "valid": 0, "total": 0}
    for label in LABELS:
        cells = list(out["cells"].get(label, {}).values())
        valid = [c for c in cells if c["valid"]]
        s = {"n_cells": len(cells), "n_valid": len(valid)}
        for k in ("H1", "H2", "H3"):
            base = valid if k == "H3" else cells   # H1/H2 over all cells; H3 only where regime valid
            have = [c for c in base if k in c]
            p = sum(1 for c in have if c[k]["pass"])
            s[k] = f"{p}/{len(have)}"
            pooled[k][0] += p; pooled[k][1] += len(have)
        pooled["valid"] += len(valid); pooled["total"] += len(cells)
        out["summary"][label] = s
        print(f"  {label:9}: valid {s['n_valid']}/{s['n_cells']}  |  "
              f"H1 {s['H1']}   H2 {s['H2']}   H3 {s['H3']} (of valid)")
    out["summary"]["pooled"] = {
        "valid": f"{pooled['valid']}/{pooled['total']}",
        "H1": f"{pooled['H1'][0]}/{pooled['H1'][1]}",
        "H2": f"{pooled['H2'][0]}/{pooled['H2'][1]}",
        "H3": f"{pooled['H3'][0]}/{pooled['H3'][1]}"}
    ps = out["summary"]["pooled"]
    print(f"  {'POOLED':9}: valid {ps['valid']}  |  H1 {ps['H1']}   H2 {ps['H2']}   H3 {ps['H3']} (of valid)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[OK] {OUT}")
    return out


if __name__ == "__main__":
    run()
