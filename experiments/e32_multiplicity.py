"""E32 — Multiplicity (Benjamini-Hochberg FDR) over the 40-cell generalization grid. ZERO-GPU.

The 40 cells = 5 ensembles x 8 benchmarks. For each of THREE pre-specified families we report the raw
per-cell significance count (bootstrap 95% CI excludes the null) AND the BH-FDR-adjusted count at q=0.05:

  H1        extinction:               dCR,                      null = 0
  H2        provenance -> correctness: n_independent dir. AUC,   null = 0.5
  selector  downstream utility:        prov_logistic_xfit dAcc,  null = 0   (this is the OLD H3 =
            selector accuracy gain; it is NOT the origin-beyond-magnitude headline, which is tested
            pooled/per-benchmark in E26, not on this grid).

Per-cell two-sided p-values are a normal approximation from the stored bootstrap CI
(SE = CI_width / (2 x 1.96)); the multiplicity check is a robustness/transparency analysis, while the
PRIMARY inference remains the pooled/global H1-H3 tests. Cells are read verbatim from the frozen verdict
tables (e25_generalization / e25_newbench / e25_scale) — nothing is regenerated.

  CUDA_VISIBLE_DEVICES="" python experiments/e32_multiplicity.py
"""
import os, sys, json
from math import erf, sqrt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

Z = 1.959964
TAB = "results/tables"
OUT = f"{TAB}/e32_multiplicity.json"


def _sf(z):                       # one-tailed standard-normal survival
    return 0.5 * (1.0 - erf(abs(z) / sqrt(2.0)))


def p_two(est, lo, hi, null):
    se = (hi - lo) / (2.0 * Z)
    if se <= 0:
        return 1.0                # degenerate cell (zero-width CI) -> no evidence
    return 2.0 * _sf((est - null) / se)


def bh(pvals, q=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank / m * q:
            kmax = rank
    passed = [False] * m
    for rank, i in enumerate(order, start=1):
        if rank <= kmax:
            passed[i] = True
    return passed


def collect_cells():
    """Return list of dicts {ens, bench, H1(est,lo,hi), H2(est,lo,hi), sel(est,lo,hi)} — one per cell."""
    cells = {}

    def add(ens, bench, cell):
        if "H1" not in cell or "H2" not in cell:
            return
        h1 = cell["H1"]; h2 = cell["H2"]; h3 = cell.get("H3", {})
        ni = h2.get("n_independent", {})
        sel = h3.get("prov_logistic_xfit", {})
        rec = {"ens": ens, "bench": bench}
        rec["H1"] = (h1["dCR"], h1["dCR_ci"][0], h1["dCR_ci"][1])
        if "dir_auc" in ni:
            rec["H2"] = (ni["dir_auc"], ni["dir_ci"][0], ni["dir_ci"][1])
        if "delta" in sel:
            rec["sel"] = (sel["delta"], sel["ci"][0], sel["ci"][1])
        cells[(ens, bench)] = rec

    g = json.load(open(f"{TAB}/e25_generalization.json"))["cells"]
    for ens, benches in g.items():
        for bench, cell in benches.items():
            add(ens, bench, cell)
    nb = json.load(open(f"{TAB}/e25_newbench.json"))["cells"]
    for bench, labels in nb.items():
        for label, cell in labels.items():
            add(label.replace("E_orig3b", "E_orig"), bench, cell)
    sc = json.load(open(f"{TAB}/e25_scale.json"))["cells"]
    for ens, benches in sc.items():
        for bench, cell in benches.items():
            add(ens, bench, cell)
    return list(cells.values())


def analyze(cells, family, null):
    rows = [c for c in cells if family in c]
    est = [c[family] for c in rows]
    raw = [lo > null for (_, lo, hi) in est]                       # CI excludes null (one-sided direction)
    pv = [p_two(e, lo, hi, null) for (e, lo, hi) in est]
    bh_pass = bh(pv, 0.05)
    # BH significance is directional: require the raw direction (est>null) AND survive BH
    bh_sig = [bh_pass[i] and est[i][0] > null for i in range(len(rows))]
    dropped = [f"{rows[i]['ens']}/{rows[i]['bench']} (p={pv[i]:.3f})"
               for i in range(len(rows)) if raw[i] and not bh_sig[i]]
    return {"n_cells": len(rows), "raw_sig": sum(raw), "bh_sig": sum(bh_sig),
            "dropped_by_bh": dropped,
            "per_cell": [{"ens": rows[i]["ens"], "bench": rows[i]["bench"],
                          "est": round(est[i][0], 4), "ci": [round(est[i][1], 4), round(est[i][2], 4)],
                          "p_two": round(pv[i], 4), "raw_sig": raw[i], "bh_sig": bh_sig[i]}
                         for i in range(len(rows))]}


def run():
    cells = collect_cells()
    res = {"n_cells_total": len(cells), "q": 0.05, "note": "two-sided normal-approx p from bootstrap CI; "
           "primary inference is pooled/global H1-H3, this grid is generalization/heterogeneity",
           "families": {}}
    for fam, null, label in [("H1", 0.0, "extinction dCR>0"),
                             ("H2", 0.5, "provenance dir-AUC>0.5"),
                             ("sel", 0.0, "selector dAcc>0 (old H3, NOT origin-beyond-magnitude)")]:
        a = analyze(cells, fam, null)
        a["label"] = label
        res["families"][fam] = a
        print(f"{fam:4} [{label}]: {a['n_cells']} cells | raw sig {a['raw_sig']}/{a['n_cells']} "
              f"-> BH-FDR sig {a['bh_sig']}/{a['n_cells']}")
        if a["dropped_by_bh"]:
            print("       dropped by BH:", "; ".join(a["dropped_by_bh"]))

    # selector-variant reconciliation: the paper's "15/40" counts EITHER frozen variant significant;
    # the primary cross-fit logistic alone is 14/40 (used above for BH).
    g = json.load(open(f"{TAB}/e25_generalization.json"))["cells"]
    def _cells3():
        acc = {}
        for e, bs in g.items():
            for b, c in bs.items(): acc[(e, b)] = c.get("H3", {})
        for b, ls in json.load(open(f"{TAB}/e25_newbench.json"))["cells"].items():
            for l, c in ls.items(): acc[(l.replace("E_orig3b", "E_orig"), b)] = c.get("H3", {})
        for e, bs in json.load(open(f"{TAB}/e25_scale.json"))["cells"].items():
            for b, c in bs.items(): acc[(e, b)] = c.get("H3", {})
        return acc
    h3 = _cells3()
    def sig(v, key):
        ci = v.get(key, {}).get("ci"); return bool(ci and ci[0] is not None and ci[0] > 0)
    ew = sum(sig(v, "prov_equalweight") for v in h3.values())
    lg = sum(sig(v, "prov_logistic_xfit") for v in h3.values())
    either = sum(sig(v, "prov_equalweight") or sig(v, "prov_logistic_xfit") for v in h3.values())
    res["selector_variants"] = {"equalweight_sig": ew, "logistic_xfit_sig": lg, "either_variant_sig": either,
                                "note": "paper's 15/40 = either_variant; primary logistic = 14/40 (BH -> 8/40)"}
    print(f"selector variants: equalweight {ew}/40, logistic {lg}/40, either {either}/40 (paper cites either=15)")
    os.makedirs(TAB, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[OK] {OUT}  (total {len(cells)} cells)")
    return res


if __name__ == "__main__":
    run()
