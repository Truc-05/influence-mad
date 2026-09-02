"""Precompute the full 40-cell derived table for the figures (zero-GPU, frozen machinery).
Pulls existing cells from the e25 verdict JSONs; fills the 6 missing base-ensemble × {bbh,truthfulqa}
cells by running the SAME frozen e18/e22/e23 verdicts on their logs. Caches -> results/tables/fig_cells.json.
"""
import os, sys, json, glob
_ROOT="/home/alex/venvs/manew/influence-mad"; sys.path.insert(0,_ROOT); os.chdir(_ROOT)
import numpy as np
import experiments.e18_hypothesis_preservation as e18
import experiments.e22_provenance as e22
import experiments.e23_provenance_selector as e23
e23.B_BOOT=800  # speed for the fill-in cells

CORE=["gsm8k","strategyqa","commonsenseqa","bbh","truthfulqa"]; NEW=["aqua","mmlu","arc"]
BENCHES=CORE+NEW; CONFIGS=["E_orig","E_7b","E_mixed","E_strong","E_n5"]
def load(p):
    try: return json.load(open(p))
    except Exception: return None
GEN=load("results/tables/e25_generalization.json"); NB=load("results/tables/e25_newbench.json"); SC=load("results/tables/e25_scale.json")
def logdir(cfg,b):
    if cfg=="E_orig": return "data/debate_logs" if (b in CORE or b=="aqua") else "data/debate_logs/gen/E_orig3b"
    return f"data/debate_logs/gen/{cfg}"

def from_json(cfg,b):
    c=None
    if cfg in ("E_orig","E_7b","E_mixed"):
        c=(GEN["cells"].get(cfg,{}).get(b) if b in CORE else NB["cells"].get(b,{}).get(cfg))
    else:
        c=SC["cells"].get(cfg,{}).get(b)
    if not c or "H1" not in c: return None
    h1,h2,h3=c["H1"],c.get("H2",{}),c.get("H3",{})
    eq=h3.get("prov_equalweight",{}); lg=h3.get("prov_logistic_xfit",{})
    return {"dCR":h1["dCR"],"h1":bool(h1["pass"]),"maj":h1["maj_acc"],"crh":h1["CR_history"],
            "auc":h2.get("n_independent",{}).get("dir_auc",float("nan")),"combo":h2.get("combo_auc",float("nan")),
            "h2":bool(h2.get("pass",False)),"fmaj":h3.get("final_majority",float("nan")),
            "delta":max(eq.get("delta",-9),lg.get("delta",-9)),"h3":bool(h3.get("pass",False))}

def compute(cfg,b):
    ld=logdir(cfg,b)
    if not glob.glob(f"{ld}/{b}_seed*.jsonl"): return None
    r18=e18.rows_for(b,logdir=ld); s18=e18.summarize(r18,f"{cfg}/{b}")
    cr=e22.candidates_for(b,logdir=ld); s22=e22.summarize(cr,f"{cfg}/{b}")
    ni=s22["signals"]["n_independent"]
    deb=e23.build_debates(benches=[b],logdir=ld); prov=e23.add_provenance_picks(deb)
    s23=e23.summarize(deb,f"{cfg}/{b}",["final_majority","round0_majority"]+prov)
    eq,lg=s23["prov_equalweight"],s23["prov_logistic_xfit"]
    return {"dCR":s18["dCR"],"h1":bool(s18["dCR_ci"][0]>0),"maj":s18["maj_acc_mode"],"crh":s18["CR_history"],
            "auc":ni["dir_auc"],"combo":s22["untuned_combo_auc"],"h2":bool(ni["dir_ci"][0]>0.5),
            "fmaj":s23["final_majority"]["acc"],"delta":max(eq["delta"],lg["delta"]),
            "h3":bool(eq["ci"][0]>0 or lg["ci"][0]>0)}

out={}
for cfg in CONFIGS:
    out[cfg]={}
    for b in BENCHES:
        c=from_json(cfg,b)
        if c is None:
            print(f"[compute] {cfg}/{b}"); c=compute(cfg,b)
        else:
            print(f"[json]    {cfg}/{b}")
        if c: out[cfg][b]=c
json.dump(out,open("results/tables/fig_cells.json","w"),indent=1)
n=sum(len(v) for v in out.values()); print(f"[OK] results/tables/fig_cells.json  ({n} cells)")
