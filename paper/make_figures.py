"""Render paper figures to results/figures/  (dpi 500, inline 'a)' captions, homogeneous panels).
Fig 1 (framework schematic) intentionally skipped.  Zero-GPU.
  CUDA_VISIBLE_DEVICES="" python paper/make_figures.py
"""
import os, sys, glob, json, warnings
from collections import defaultdict
import numpy as np
warnings.filterwarnings("ignore")
_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..")); sys.path.insert(0,_ROOT); os.chdir(_ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import experiments.e22_provenance as e22

FIGDIR="results/figures"; os.makedirs(FIGDIR,exist_ok=True); DPI=500
# --- blue-anchored, colourblind-safe palette (validated: worst adjacent CVD ΔE 16.2 ≥ 12;
#     every chart carries direct value labels, which satisfies the aqua relief rule) ---
IND ="#2a78d6"   # independent origin / provenance selector — the hero (blue)
INH ="#eb6834"   # inherited / socially-adopted support (orange)
CORR="#1baf7a"   # correct / provenance-positive (aqua-green)
HARM="#e34948"   # harmful / forgotten / extinction (red)
HIST="#4a3aa7"   # history pool / oracle ceiling (violet)
BASE="#9aa1ab"   # baseline / deployed / neutral (gray)
ACC ="#6da7ec"   # secondary blue accent (round-0, hazard)
BROWN="#937860"  # extra series
INK ="#15181d"   # primary ink
MUT ="#6b7280"   # muted axis / chance-line ink
from matplotlib.colors import LinearSegmentedColormap as _LSC
CM_LOSS=_LSC.from_list("loss",["#fbe9e8","#f4a6a4","#e34948","#a92826"])  # H1 extinction (red seq)
CM_GOOD=_LSC.from_list("good",["#dcf3ea","#83d7b4","#1baf7a","#0a7350"])  # H2 provenance (aqua seq)
CM_BLUE=_LSC.from_list("blue",["#e6effb","#9ec5f4","#3987e5","#123f7a"])  # H3 / selector (blue seq)
CM_DIV =_LSC.from_list("div", ["#a92826","#e8615f","#f2f1ee","#5598e7","#1b4f8f"])  # diverging red↔blue
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,"savefig.dpi":DPI,"figure.dpi":120,
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#cbd0d8","axes.linewidth":1.0,
    "axes.labelcolor":INK,"text.color":INK,"xtick.color":MUT,"ytick.color":MUT,"axes.titlecolor":INK,
    "xtick.labelcolor":INK,"ytick.labelcolor":INK,
    "figure.facecolor":"white","axes.facecolor":"white","axes.axisbelow":True,
    "axes.grid":True,"grid.color":"#e3e7ee","grid.linewidth":1.0,
    "axes.titlesize":12,"axes.titlepad":9,"axes.titleweight":"regular"})

CORE=["gsm8k","strategyqa","commonsenseqa","bbh","truthfulqa"]; NEW=["aqua","mmlu","arc"]
BENCHES=CORE+NEW; CONFIGS=["E_orig","E_7b","E_mixed","E_strong","E_n5"]
BLAB={"gsm8k":"GSM8K","strategyqa":"StrategyQA","commonsenseqa":"CSQA","bbh":"BBH","truthfulqa":"TQA",
      "aqua":"AQuA","mmlu":"MMLU","arc":"ARC"}
def load(p):
    try: return json.load(open(p))
    except Exception: return None
E18=load("results/tables/e18_hypothesis_preservation.json"); E22=load("results/tables/e22_provenance.json")
E26=load("results/tables/e26_independence.json"); E2=load("results/tables/e2_mechanism_influence.json")
E28=load("results/tables/e28_h3_robust.json")
CELLS=load("results/tables/fig_cells.json")

def logdir(cfg,b):
    if cfg=="E_orig": return "data/debate_logs" if (b in CORE or b=="aqua") else "data/debate_logs/gen/E_orig3b"
    return f"data/debate_logs/gen/{cfg}"
print("building candidate table ...")
CAND=[]
for cfg in CONFIGS:
    for b in BENCHES:
        ld=logdir(cfg,b)
        if not glob.glob(f"{ld}/{b}_seed*.jsonl"): continue
        for r in e22.candidates_for(b,logdir=ld):
            r["cfg"]=cfg; CAND.append(r)
print(f"  {len(CAND)} candidates")

# ---------- helpers ----------
def T(ax,tag,text,size=12): ax.set_title(f"{tag}) {text}",loc="left",fontsize=size,color=INK,fontweight="regular")
def wilson(p,n,z=1.96):
    if n==0: return p,p
    d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def forest(ax,labels,vals,los,his,color=IND,ref=0.0,xlabel="",star=None,dot=6):
    y=np.arange(len(labels))[::-1]
    ax.axvline(ref,color=BASE,lw=1.1,ls="--",zorder=1)
    for yi,v,lo,hi in zip(y,vals,los,his):
        ax.plot([lo,hi],[yi,yi],color=color,lw=2.4,zorder=2,solid_capstyle="round")
        ax.plot([v],[yi],"o",color=color,ms=dot,zorder=3,mec="white",mew=1.2)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel(xlabel); ax.margins(y=.08)
    if star:
        xr=ax.get_xlim()[1]
        for yi,s in zip(y,star):
            if s: ax.annotate("∗",(xr,yi),fontsize=13,color=color,va="center",fontweight="bold")
import matplotlib.colors as _mc
def _rgb(c): return np.array(_mc.to_rgb(c))
def _lighten(c,f=0.42): return _mc.to_hex(tuple((1-f)*_rgb(c)+f*np.ones(3)))
def _darken(c,f=0.18):  return _mc.to_hex(tuple((1-f)*_rgb(c)))
def _desat(c,f=0.7):
    import colorsys; r,g,b=_rgb(c); h,l,s=colorsys.rgb_to_hls(r,g,b)
    return _mc.to_hex(colorsys.hls_to_rgb(h,min(0.9,l+0.12),s*(1-f)))
def _tc(cmap,v,vmin,vmax):
    """Readable text colour on a heatmap cell: white on dark fills, ink on light ones (luminance)."""
    cm=plt.get_cmap(cmap) if isinstance(cmap,str) else cmap
    t=0.5 if vmax<=vmin else max(0.,min(1.,(v-vmin)/(vmax-vmin)))
    r,g,b,_=cm(t); return "white" if (0.299*r+0.587*g+0.114*b)<0.62 else INK
def gbar(ax,x,h,w,color,z=3,edge="white",elw=1.1,base=0.0):
    """Modern vertical-gradient bar: deep at the base, brighter toward the tip."""
    cmap=_mc.LinearSegmentedColormap.from_list("g",[_darken(color,0.14),_lighten(color,0.40)])
    lo,hi=min(base,h),max(base,h)
    ax.imshow(np.linspace(0,1,256).reshape(-1,1),extent=(x-w/2,x+w/2,lo,hi),origin="lower",
              aspect="auto",cmap=cmap,vmin=0,vmax=1,zorder=z,interpolation="bilinear")
    if edge: ax.add_patch(plt.Rectangle((x-w/2,lo),w,hi-lo,fill=False,ec=edge,lw=elw,zorder=z+0.4))
def heat(ax,M,xt,yt,cmap,vmin,vmax,fmt="{:.2f}",thr=None):
    """Seamless (borderless) heatmap with high-contrast bold labels."""
    n0,n1=M.shape
    im=ax.imshow(M,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto",interpolation="nearest"); ax.grid(False)
    for i in range(n0):
        for j in range(n1):
            if not np.isnan(M[i,j]):
                ax.text(j,i,fmt.format(M[i,j]),ha="center",va="center",fontsize=8.4,zorder=4,fontweight="bold",
                        color=_tc(cmap,M[i,j],vmin,vmax))
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt,fontsize=8.6,rotation=35,ha="right")
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt,fontsize=9.2)
    return im
def save(fig,name,pad=.4):
    fig.tight_layout(rect=[0,0,1,.965],pad=pad,w_pad=1.8,h_pad=2.2); p=f"{FIGDIR}/{name}.png"
    fig.savefig(p,bbox_inches="tight",facecolor="white"); plt.close(fig); print("  saved",p)
def cell(cfg,b): return CELLS.get(cfg,{}).get(b)

# =========================================================================
def fig2():  # extinction — 3 homogeneous horizontal per-benchmark panels
    fig,axs=plt.subplots(1,3,figsize=(16,5.0))
    order=["gsm8k","strategyqa","commonsenseqa","bbh","truthfulqa"]; y=np.arange(len(order))[::-1]
    ax=axs[0]
    for yi,b in zip(y,order):
        d=E18[b]; ax.plot([d["CR_final"],d["CR_history"]],[yi,yi],color=HIST,lw=3,zorder=1,solid_capstyle="round")
        ax.plot([d["CR_final"]],[yi],"o",color=BASE,ms=9,zorder=2)
        ax.plot([d["CR_history"]],[yi],"o",color=HIST,ms=9,zorder=2)
    ax.plot([],[],"o",color=BASE,label="final pool"); ax.plot([],[],"o",color=HIST,label="history pool")
    ax.set_yticks(y); ax.set_yticklabels([BLAB[b] for b in order]); ax.set_xlim(0,1); ax.set_xlabel("P(correct answer in pool)")
    ax.legend(loc="lower right",frameon=False,fontsize=9.5); T(axs[0],"a","preservation recovers the forgotten answer")
    ax=axs[1]
    forest(ax,[BLAB[b] for b in order],[E18[b]["dCR"] for b in order],
           [E18[b]["dCR_ci"][0] for b in order],[E18[b]["dCR_ci"][1] for b in order],color=HARM,ref=0,
           xlabel="ΔCR = forgotten-correct rate  [95% CI]"); T(ax,"b","how much correctness is forgotten")
    ax=axs[2]
    for yi,b in zip(y,order):
        ax.plot([E18[b]["pool_final"],E18[b]["pool_hist"]],[yi,yi],color="#cfd4db",lw=3,zorder=1,solid_capstyle="round")
        ax.plot([E18[b]["pool_final"]],[yi],"o",color=BASE,ms=9); ax.plot([E18[b]["pool_hist"]],[yi],"o",color=HIST,ms=9)
    ax.set_yticks(y); ax.set_yticklabels([BLAB[b] for b in order]); ax.set_xlabel("distinct hypotheses / question")
    ax.plot([],[],"o",color=BASE,label="final pool"); ax.plot([],[],"o",color=HIST,label="history pool")
    ax.legend(loc="lower right",frameon=False,fontsize=9.5); T(ax,"c","preserving is cheap (pool ≈ 3)")
    fig.suptitle("Correct hypotheses are present, then socially forgotten",x=.012,ha="left",fontsize=14,fontweight="bold")
    save(fig,"fig2_extinction")

def fig3():  # reliability curves — 4 identical-axis panels
    fig,axs=plt.subplots(2,2,figsize=(12,8.8)); axs=axs.ravel(); P=E22["pooled"]
    def curve(ax,sig,color,xlabel):
        bins=P["signals"][sig]["bins"]; xs=sorted(int(k) for k in bins)
        ys=[bins[str(x)][0] for x in xs]; ns=[bins[str(x)][1] for x in xs]
        lo=[wilson(v,n)[0] for v,n in zip(ys,ns)]; hi=[wilson(v,n)[1] for v,n in zip(ys,ns)]
        ax.fill_between(xs,lo,hi,color=color,alpha=.16); ax.plot(xs,ys,"-o",color=color,lw=2.4,ms=8,mec="white",mew=1.4)
        for x,v in zip(xs,ys): ax.annotate(f"{v:.2f}",(x,v),textcoords="offset points",xytext=(0,10),ha="center",fontsize=8.5,color=color)
        ax.set_xticks(xs); ax.set_ylim(0,1); ax.set_xlabel(xlabel); ax.set_ylabel("P(correct)")
    curve(axs[0],"n_independent",IND,"n_independent (round-0 origin)"); T(axs[0],"a","independent origin → correct")
    curve(axs[1],"n_persist",HIST,"n_persist (rounds alive)"); T(axs[1],"b","persistence → correct")
    curve(axs[2],"first_round",HARM,"first round seen"); T(axs[2],"c","late emergence → wrong")
    ax=axs[3]
    for sig,c,lb in [("n_late_adopt",BASE,"late_adopt"),("resurrected",INH,"resurrected")]:
        bins=P["signals"][sig]["bins"]; xs=sorted(int(k) for k in bins); ys=[bins[str(x)][0] for x in xs]
        ax.plot(xs,ys,"-o",color=c,lw=2.2,ms=7,label=lb)
    ax.axhline(P["base_rate"],color=MUT,ls=":",lw=1.3,label="base rate"); ax.set_ylim(0,1)
    ax.set_xlabel("feature value"); ax.set_ylabel("P(correct)"); ax.legend(frameon=False,fontsize=9)
    T(axs[3],"d","null features (≈ chance)")
    fig.suptitle("A hypothesis's provenance predicts its correctness",x=.012,ha="left",fontsize=14,fontweight="bold")
    save(fig,"fig3_reliability")

def fig4():  # feature importance — 2 horizontal panels
    fig,axs=plt.subplots(1,2,figsize=(13,5.2)); P=E22["pooled"]
    sf=["n_support","n_independent","n_persist","first_round","n_late_adopt","resurrected"]
    vals=[P["signals"][s]["dir_auc"] for s in sf]; los=[P["signals"][s]["dir_ci"][0] for s in sf]; his=[P["signals"][s]["dir_ci"][1] for s in sf]
    o=np.argsort(vals); sf=[sf[i] for i in o]; vals=[vals[i] for i in o]; los=[los[i] for i in o]; his=[his[i] for i in o]
    cols=[IND if s in("n_independent","n_persist") else (BASE if s in("n_late_adopt","resurrected") else ACC) for s in sf]
    ax=axs[0]; yy=np.arange(len(sf))[::-1]; ax.axvline(.5,color=BASE,ls="--",lw=1.1)
    for yi,v,lo,hi,c in zip(yy,vals,los,his,cols):
        ax.plot([lo,hi],[yi,yi],color=c,lw=2.4,solid_capstyle="round"); ax.plot([v],[yi],"o",color=c,ms=6,mec="white",mew=1.2)
    ax.set_yticks(yy); ax.set_yticklabels([s.replace("n_","") for s in sf]); ax.set_xlabel("directional AUC  [95% CI]  (0.5 = chance)")
    T(ax,"a","which features predict correctness")
    ax=axs[1]
    feats=["n_independent","n_support","n_persist","n_late_adopt"]; flab=["indep origin","total support","persist","late adopt"]
    cor=[r for r in CAND if r["correct"]]; wr=[r for r in CAND if not r["correct"]]
    mc=[np.mean([r[f] for r in cor]) for f in feats]; mw=[np.mean([r[f] for r in wr]) for f in feats]
    mx=[max(a,b) for a,b in zip(mc,mw)]; mc=[a/m for a,m in zip(mc,mx)]; mw=[a/m for a,m in zip(mw,mx)]
    yy=np.arange(len(feats))[::-1]; h=.36
    ax.barh(yy+h/2,mc,h,color=CORR,ec="white",lw=.8,label="correct hypothesis"); ax.barh(yy-h/2,mw,h,color=HARM,ec="white",lw=.8,label="wrong hypothesis")
    ax.set_yticks(yy); ax.set_yticklabels(flab); ax.set_xlabel("mean feature value (normalized)"); ax.set_xlim(0,1.15)
    y0,y1=ax.get_ylim(); ax.set_ylim(y0,y1+(y1-y0)*0.15)
    ax.legend(fontsize=9.5,loc="upper center",ncol=2,frameon=False)
    ax.grid(axis="y",visible=False); T(ax,"b","correct vs wrong fingerprint")
    fig.suptitle("Which provenance signals carry correctness information",x=.012,ha="left",fontsize=14,fontweight="bold")
    save(fig,"fig4_features")

def fig5():  # HERO matched support — 2 panels
    fig,axs=plt.subplots(1,2,figsize=(13.6,5.6)); ms={int(s):{int(i):v for i,v in row.items()} for s,row in E26["matched_stratum"].items()}
    svals=[s for s in sorted(ms) if len(ms[s])>=2][:7]; ivals=[0,1,2,3]
    M=np.full((len(svals),len(ivals)),np.nan)
    for i,s in enumerate(svals):
        for j,iv in enumerate(ivals):
            if iv in ms[s]: M[i,j]=ms[s][iv][0]
    ax=axs[0]; im=heat(ax,M,ivals,svals,CM_BLUE,0,1,fmt="{:.2f}",thr=.55)
    ax.set_xlabel("independent origin  $n_{ind}$   →"); ax.set_ylabel("total support  $n_{sup}$")
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03); cb.set_label("P(correct)",fontsize=9); cb.ax.tick_params(labelsize=8)
    T(ax,"a","P(correct) rises along each row (fixed total support)")
    ax=axs[1]; cmap=plt.cm.viridis(np.linspace(.06,.86,len(svals)))
    for c,s in zip(cmap,svals):
        pts=sorted(ms[s].items()); xs=[i for i,_ in pts]; ys=[v[0] for _,v in pts]
        ax.plot(xs,ys,"-o",color=c,lw=2.2,ms=6,label=str(s))
    ax.set_xticks([0,1,2,3]); ax.set_ylim(0,1); ax.set_xlabel("independent origin  $n_{ind}$"); ax.set_ylabel("P(correct)")
    ax.legend(title="total support  $n_{sup}$",frameon=False,fontsize=8.5,ncol=2,title_fontsize=8.5,loc="lower right")
    T(ax,"b","every support stratum slopes up")
    fig.suptitle("★  Not all consensus is equal: at matched total support, independent origin raises correctness",
                 x=.012,ha="left",fontsize=13.5,fontweight="bold")
    save(fig,"fig5_matched_support_HERO")

def fig6():  # coefficients — 4 forest panels (homogeneous)
    fig,axs=plt.subplots(2,2,figsize=(12.5,9.0)); axs=axs.ravel()
    # a) TRUE decomposition: independent origin vs GENUINE adoption vs persistence (E28)
    dec=E28["true_decomposition"]; bi=dec["n_independent"]; ba=dec["n_adopt"]; bp=dec["n_persist"]
    forest(axs[0],["independent\norigin","social\nadoption","persistence\n(per round)"],
           [bi["beta"],ba["beta"],bp["beta"]],[bi["lo"],ba["lo"],bp["lo"]],[bi["hi"],ba["hi"],bp["hi"]],
           color=IND,ref=0,xlabel="β per unit (log-odds)")
    axs[0].plot([ba["beta"]],[1],"o",color=INH,ms=7,mec="white",mew=1.2,zorder=4)     # adoption in orange
    axs[0].plot([ba["lo"],ba["hi"]],[1,1],color=INH,lw=2.4,solid_capstyle="round",zorder=3)
    d=dec["contrast_ind_minus_adopt"]
    axs[0].annotate(f"Δβ = {d['beta']:.2f}",(bi['beta']*.52,1.5),fontsize=14,fontweight="bold",color=HARM)
    T(axs[0],"a",f"independent origin ≫ social adoption (Δβ = {d['beta']:.2f})")
    # b) beta_ind survives controls INCLUDING within-question fixed effects
    A=E26["model_A_ctrl_support"]["n_independent"]; F=E26["model_A_fixed_effects"]["n_independent"]
    W=E28["within_question_condlogit_ctrl_support"]["n_independent"]
    FS=E28["robust_ctrl_final_support"]["n_independent"]; RD=E28["robust_ctrl_rounds"]["n_independent"]
    rungs=[("matched cumul.\nsupport",A),("matched FINAL\nsupport",FS),("+ bench & config\nFE",F),
           ("+ within-question\nFE",W),("+ exposure\nrounds",RD)]
    labs=[l for l,_ in rungs]; vals=[d["beta"] for _,d in rungs]
    los=[d["lo"] for _,d in rungs]; his=[d["hi"] for _,d in rungs]
    forest(axs[1],labs,vals,los,his,color=IND,ref=0,xlabel="β  (independent | controls)")
    T(axs[1],"b","survives every control (incl. final-vote consensus)")
    # c per-benchmark
    pb=E26["per_benchmark"]; o=sorted(BENCHES,key=lambda b:-pb[b]["beta"])
    forest(axs[2],[BLAB[b] for b in o],[pb[b]["beta"] for b in o],[pb[b]["lo"] for b in o],[pb[b]["hi"] for b in o],
           color=IND,ref=0,xlabel="β  (independent | total support)",star=[pb[b]["lo"]>0 for b in o]); T(axs[2],"c","7 / 8 benchmarks hold")
    # d per-config
    pc=E26["per_config"]; o=["E_orig","E_7b","E_mixed","E_strong","E_n5"]
    forest(axs[3],o,[pc[c]["beta"] for c in o],[pc[c]["lo"] for c in o],[pc[c]["hi"] for c in o],
           color=IND,ref=0,xlabel="β",star=[pc[c]["lo"]>0 for c in o]); T(axs[3],"d","5 / 5 ensembles (incl. 14B, N=5)")
    fig.suptitle("Independent origin carries correctness signal beyond magnitude — and question difficulty",
                 x=.012,ha="left",fontsize=14,fontweight="bold")
    save(fig,"fig6_coefficients")

def fig7():  # generalization — 3 aligned heatmaps
    fig,axs=plt.subplots(1,3,figsize=(16.5,5.0)); xt=[BLAB[b] for b in BENCHES]
    M1=np.array([[cell(c,b)["dCR"] if cell(c,b) else np.nan for b in BENCHES] for c in CONFIGS])
    M2=np.array([[cell(c,b)["auc"] if cell(c,b) else np.nan for b in BENCHES] for c in CONFIGS])
    b3=np.full((len(CONFIGS),len(BENCHES)),np.nan)
    for i,c in enumerate(CONFIGS):
        for j,b in enumerate(BENCHES):
            rows=[r for r in CAND if r["cfg"]==c and r["bench"]==b]
            if len(rows)>20 and len(set(r["correct"] for r in rows))>1:
                X=np.array([[r["n_independent"],r["n_support"]] for r in rows],float)
                b3[i,j]=LogisticRegression(C=1e6,max_iter=2000).fit(X,[r["correct"] for r in rows]).coef_[0][0]
    heat(axs[0],M1,xt,CONFIGS,CM_LOSS,0,.22,fmt="{:.2f}"); T(axs[0],"a","H1  extinction  (ΔCR)")
    heat(axs[1],M2,xt,CONFIGS,CM_GOOD,.5,.95,fmt="{:.2f}"); T(axs[1],"b","H2  provenance → correct  (AUC)")
    heat(axs[2],b3,xt,CONFIGS,CM_BLUE,0,1.3,fmt="{:.2f}"); T(axs[2],"c","H3  independent > inherited  (β)")
    fig.suptitle("Generalization: the two scientific claims travel across models, tasks and agent counts",
                 x=.012,ha="left",fontsize=14,fontweight="bold")
    save(fig,"fig7_generalization")

def fig8():  # utility — heatmap + 2 scatters
    fig=plt.figure(figsize=(17,5.0)); gs=GridSpec(1,3,figure=fig,width_ratios=[1.12,1,1],wspace=.24)
    axH=fig.add_subplot(gs[0,0])
    M=np.full((len(CONFIGS),len(BENCHES)),np.nan); S=np.zeros_like(M)
    for i,c in enumerate(CONFIGS):
        for j,b in enumerate(BENCHES):
            cc=cell(c,b)
            if cc: M[i,j]=cc["delta"]; S[i,j]=cc["h3"]
    im=axH.imshow(M,cmap=CM_DIV,vmin=-.16,vmax=.16,aspect="auto",interpolation="nearest"); axH.grid(False)
    for i in range(len(CONFIGS)):
        for j in range(len(BENCHES)):
            if not np.isnan(M[i,j]):
                axH.text(j,i,("∗" if S[i,j] else "")+f"{M[i,j]:+.02f}",ha="center",va="center",fontsize=7.5,zorder=4,
                         fontweight="bold",color=_tc(CM_DIV,M[i,j],-.16,.16))
    axH.set_xticks(range(len(BENCHES))); axH.set_xticklabels([BLAB[b] for b in BENCHES],fontsize=8,rotation=35,ha="right")
    axH.set_yticks(range(len(CONFIGS))); axH.set_yticklabels(CONFIGS,fontsize=9)
    T(axH,"a","selector ΔAcc  (∗ = sig, 14/40)")
    def scat(ax,xf,xlab,tag,tit):
        xs=[];ys=[];cs=[]
        for c in CONFIGS:
            for b in BENCHES:
                cc=cell(c,b)
                if not cc or np.isnan(cc["delta"]): continue
                xs.append(xf(cc)); ys.append(cc["delta"]); cs.append(CORR if cc["h3"] else BASE)
        ax.axhline(0,color=BASE,ls="--",lw=1.1); ax.scatter(xs,ys,c=cs,s=42,ec="white",lw=.7)
        ax.set_xlabel(xlab); ax.set_ylabel("selector ΔAcc"); T(ax,tag,tit)
    scat(fig.add_subplot(gs[0,1]),lambda c:c["crh"]-c["fmaj"],"headroom (CR_hist − majority)","b","gain needs headroom")
    scat(fig.add_subplot(gs[0,2]),lambda c:c["combo"],"provenance identifiability (AUC)","c","gain needs identifiable signal")
    fig.suptitle("From mechanism to decision utility: reranking helps only where headroom + identifiability meet",
                 x=.012,ha="left",fontsize=13.5,fontweight="bold")
    save(fig,"fig8_utility")

def fig9():  # why modest + beyond influence
    fig,axs=plt.subplots(1,3,figsize=(15.8,4.8))
    # a collinearity
    ax=axs[0]; ni=np.array([r["n_independent"] for r in CAND],float); ns=np.array([r["n_support"] for r in CAND],float)
    rng=np.random.default_rng(0); idx=rng.choice(len(ni),min(4000,len(ni)),replace=False)
    ax.scatter(ns[idx]+rng.normal(0,.12,len(idx)),ni[idx]+rng.normal(0,.12,len(idx)),s=6,alpha=.12,color=IND,ec="none")
    ax.set_xlabel("total support"); ax.set_ylabel("independent origin")
    ax.text(.04,.9,f"r = {E26['corr_ind_support']:.2f}",transform=ax.transAxes,fontsize=12,fontweight="bold")
    T(ax,"a","origin & total support are collinear")
    # b incremental AUC lollipop
    ax=axs[1]; inc=E26["incremental_auc"]; labs=["independent","total support","support +\nindependent"]; vals=[inc["n_independent"],inc["n_support"],inc["both"]]
    yy=np.arange(3)[::-1]
    for yi,v,c in zip(yy,vals,[IND,BASE,HIST]):
        ax.plot([.75,v],[yi,yi],color=c,lw=2.2); ax.plot([v],[yi],"o",color=c,ms=11,mec="white",mew=1.3)
        ax.annotate(f"{v:.3f}",(v,yi),textcoords="offset points",xytext=(9,0),va="center",fontsize=9.5)
    ax.set_yticks(yy); ax.set_yticklabels(labs); ax.set_xlim(.75,.9); ax.set_xlabel("AUC (rank candidates)")
    ax.annotate(f"Δ +{inc['delta_over_support']:.3f}",(inc['both'],.5),fontsize=10,color=HARM,fontweight="bold")
    T(ax,"b","origin adds little to ranking (Δ+.007)")
    # c influence AUC
    ax=axs[2]; au=E2["auc"]; labs=["influence\nonly","structural\ncontrol","perm.\nfloor"]; vals=[au["influence_only"],au["structural_only"],au["permutation_floor"]]
    ax.bar(range(3),vals,color=[HARM,IND,BASE],width=.62,ec="white",lw=1.2); ax.grid(axis="y")
    ax.axhline(.5,color=MUT,ls="--",lw=1.2); ax.annotate("chance",(2.4,.51),fontsize=8.5,color=MUT)
    for i,v in enumerate(vals): ax.annotate(f"{v:.2f}",(i,v),textcoords="offset points",xytext=(0,5),ha="center",fontweight="bold",fontsize=9.5,color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=9); ax.set_ylim(0,.75); ax.set_ylabel("AUC (diagnose mechanism)")
    T(ax,"c","influence ≈ chance for diagnosis")
    fig.suptitle("Why the selector is modest, and why provenance beats influence",x=.012,ha="left",fontsize=14,fontweight="bold")
    save(fig,"fig9_interpretation")

def fig10():  # OVERVIEW — small multiples, one identical panel per benchmark
    order=sorted(BENCHES,key=lambda b:-E26["per_benchmark"][b]["beta"])
    fig,axs=plt.subplots(2,4,figsize=(17,8.2)); axs=axs.ravel()
    for k,b in enumerate(order):
        ax=axs[k]; rows=[r for r in CAND if r["bench"]==b]
        y=[r["correct"] for r in rows]; ni=[r["n_independent"] for r in rows]
        xs=[0,1,2,3]; xv=[];yv=[];lo=[];hi=[]
        for v in xs:
            sub=[c for c,n in zip(y,ni) if n==v]
            if len(sub)>=10:
                p=float(np.mean(sub)); l,h=wilson(p,len(sub)); xv.append(v);yv.append(p);lo.append(l);hi.append(h)
        base=float(np.mean(y))
        ax.axhline(base,color=BASE,ls=":",lw=1.3,zorder=1)
        ax.fill_between(xv,lo,hi,color=IND,alpha=.16,zorder=2)
        ax.plot(xv,yv,"-o",color=IND,lw=2.4,ms=8,mec="white",mew=1.4,zorder=3)
        for x,p in zip(xv,yv): ax.annotate(f"{p:.2f}",(x,p),textcoords="offset points",xytext=(0,9),ha="center",fontsize=8,color=IND)
        auc=roc_auc_score(y,ni) if len(set(y))>1 else float("nan")
        bet=E26["per_benchmark"][b]["beta"]; sig=E26["per_benchmark"][b]["lo"]>0
        ax.text(.05,.95,f"β = {bet:+.2f}{'∗' if sig else ''}\nAUC = {auc:.2f}",transform=ax.transAxes,
                fontsize=9.5,va="top",ha="left",bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#d7dbe2"))
        ax.set_xticks([0,1,2,3]); ax.set_ylim(0,1); ax.set_xlim(-.25,3.25)
        T(ax,chr(97+k),BLAB[b])
        if k%4==0: ax.set_ylabel("P(correct)")
        if k>=4: ax.set_xlabel("independent origin  $n_{ind}$")
    fig.suptitle("Results at a glance — in every benchmark, more independently-produced support ⇒ more often correct",
                 x=.012,ha="left",fontsize=14,fontweight="bold")
    save(fig,"fig10_overview")

def fig10b():  # OVERVIEW v2 — 2 charts per row (4x2), wide landscape panels
    order=sorted(BENCHES,key=lambda b:-E26["per_benchmark"][b]["beta"])
    fig,axs=plt.subplots(4,2,figsize=(14,14.5)); axs=axs.ravel()
    for k,b in enumerate(order):
        ax=axs[k]; rows=[r for r in CAND if r["bench"]==b]
        y=[r["correct"] for r in rows]; ni=[r["n_independent"] for r in rows]
        xv=[];yv=[];lo=[];hi=[]
        for v in [0,1,2,3]:
            sub=[c for c,n in zip(y,ni) if n==v]
            if len(sub)>=10:
                p=float(np.mean(sub)); l,h=wilson(p,len(sub)); xv.append(v);yv.append(p);lo.append(l);hi.append(h)
        base=float(np.mean(y))
        ax.axhline(base,color=BASE,ls=":",lw=1.3,zorder=1)
        ax.fill_between(xv,lo,hi,color=IND,alpha=.16,zorder=2)
        ax.plot(xv,yv,"-o",color=IND,lw=2.6,ms=9,mec="white",mew=1.5,zorder=3)
        for x,p in zip(xv,yv): ax.annotate(f"{p:.2f}",(x,p),textcoords="offset points",xytext=(0,10),ha="center",fontsize=9,color=IND)
        auc=roc_auc_score(y,ni) if len(set(y))>1 else float("nan")
        bet=E26["per_benchmark"][b]["beta"]; sig=E26["per_benchmark"][b]["lo"]>0
        ax.text(.04,.95,f"β = {bet:+.2f}{'∗' if sig else ''}   AUC = {auc:.2f}",transform=ax.transAxes,
                fontsize=10.5,va="top",ha="left",bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#d7dbe2"))
        ax.set_xticks([0,1,2,3]); ax.set_ylim(0,1); ax.set_xlim(-.25,3.25)
        T(ax,chr(97+k),BLAB[b],size=13)
        if k%2==0: ax.set_ylabel("P(correct)")
        if k>=6: ax.set_xlabel("independent origin  $n_{ind}$")
    fig.suptitle("Results at a glance — in every benchmark, more independently-produced support ⇒ more often correct",
                 x=.012,ha="left",fontsize=14.5,fontweight="bold")
    save(fig,"fig10_overview_v2")

def fig11():  # selector ladder — grouped bars (beats baselines, ceiling remains)
    E23=load("results/tables/e23_provenance_selector.json")
    order=["gsm8k","commonsenseqa","bbh","strategyqa","truthfulqa","pooled"]
    labs=[BLAB.get(b,"Pooled") for b in order]
    sels=[("final_majority","final majority (deployed)",BASE),
          ("influence_vote","influence vote (conference)",HARM),
          ("round0_majority","round-0 majority",ACC),
          ("prov_logistic_xfit","provenance selector (ours)",IND),
          ("CR_history","oracle ceiling",HIST)]
    fig,ax=plt.subplots(figsize=(13.5,5.4)); x=np.arange(len(order)); w=.155
    for i,(key,lab,c) in enumerate(sels):
        vals=[E23[b]["CR_history"] if key=="CR_history" else E23[b][key]["acc"] for b in order]
        ax.bar(x+(i-2)*w,vals,w,color=c,label=lab,ec="white",lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=11); ax.set_ylabel("accuracy  /  P(correct in pool)")
    ax.set_ylim(0,1); ax.grid(axis="y")
    ax.legend(frameon=False,fontsize=9.5,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.14))
    fig.suptitle("A zero-cost provenance selector beats the deployed majority and the conference influence-vote",
                 x=.012,ha="left",fontsize=13.5,fontweight="bold",y=1.0)
    save(fig,"fig11_selector_ladder")

def fig12():  # failure != benefit (C2 bridge)
    E13=load("results/tables/e13_benefit_aware.json"); P=E13["pooled"]
    fig,axs=plt.subplots(1,2,figsize=(13,5.2))
    ax=axs[0]; cur=P["curves"]; never=P["never"]
    xs=np.linspace(0,1,len(cur["random"]))
    ax.axvspan(0.2,0.4,color=INH,alpha=0.07,lw=0)   # operating band
    ax.axhline(never,color=MUT,ls=":",lw=1.3,label="never")
    for key,lab,c,lw,ls in [("oracle (true ITE)","oracle",HIST,2.3,"--"),
                            ("CATE (which helps)","learned CATE",IND,2.7,"-"),
                            ("hazard (which fails)","hazard",ACC,1.9,"-"),
                            ("random","random",BASE,1.7,"-")]:
        if key in cur: ax.plot(xs,cur[key],ls=ls,color=c,lw=lw,marker="o",ms=3.2,label=lab)
    ax.set_xlabel("intervention budget  (fraction intervened) →"); ax.set_ylabel("final-answer accuracy")
    y0,y1=ax.get_ylim(); ax.set_ylim(y0,y1+(y1-y0)*0.42)
    ax.legend(frameon=False,fontsize=8.6,loc="upper center",ncol=3,columnspacing=1.5,handlelength=1.6)
    T(ax,"a","huge oracle ceiling, but learned targeting ≈ random")
    ax=axs[1]; d=P["diagnostics"]
    labs=["predict Y0\n(will it fail?)","predict Y1\n(will fix work?)","predict EFFECT\n(who benefits?)"]
    vals=[d["auc_predict_Y0"],d["auc_predict_Y1"],d["auc_predict_effect"]]
    cols=[CORR,CORR,HARM]
    ax.bar(range(3),vals,color=cols,width=.6,ec="white",lw=1.2); ax.grid(axis="y")
    ax.axhline(.5,color=MUT,ls="--",lw=1.2); ax.annotate("chance",(2.35,.51),fontsize=8.5,color=MUT)
    for i,v in enumerate(vals): ax.annotate(f"{v:.2f}",(i,v),textcoords="offset points",xytext=(0,5),ha="center",fontweight="bold",fontsize=10,color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=9.5); ax.set_ylim(0,.8); ax.set_ylabel("AUC")
    T(ax,"b","predictability ≠ actionability")
    fig.suptitle("Predicting failure is not predicting benefit: the difference of two predictable outcomes is unpredictable",
                 x=.012,ha="left",fontsize=13,fontweight="bold")
    save(fig,"fig12_failure_not_benefit")

def fig13():  # scale-extension atlas — 24B open-weight anchor + frontier API (deepseek), appendix
    SC=load("results/tables/e25_scale_ext.json")
    if not SC or not SC.get("cells"):
        print("  no e25_scale_ext.json, skip fig13"); return
    LAB=["E_strong24","E_frontier"]
    LABT=["E_strong24\n(24B open-wt)","E_frontier\n(deepseek-v4-flash)"]
    fig,axs=plt.subplots(1,3,figsize=(16.5,3.9))
    def panel(ax,tag,title,cmap,vmin,vmax,getval,getsig,thr,fmt="{:+.2f}"):
        M=np.full((len(LAB),len(BENCHES)),np.nan); Sg=np.zeros_like(M); IV=np.zeros((len(LAB),len(BENCHES)),bool)
        for i,lab in enumerate(LAB):
            for j,b in enumerate(BENCHES):
                c=SC["cells"].get(lab,{}).get(b)
                if not c: continue
                v=getval(c)
                if v is None: continue
                M[i,j]=v; Sg[i,j]=bool(getsig(c)); IV[i,j]=(not c.get("valid",True))
        ax.imshow(M,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto",interpolation="nearest"); ax.grid(False)
        for i in range(len(LAB)):
            for j in range(len(BENCHES)):
                if np.isnan(M[i,j]): continue
                if IV[i,j]:   # invalid-gate marker: light cross-hatch on the cell
                    ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,hatch="xxx",edgecolor="#7f858d",lw=.5,zorder=3))
                ax.text(j,i,("∗" if Sg[i,j] else "")+fmt.format(M[i,j]),ha="center",va="center",fontsize=8.7,zorder=6,
                        fontweight="bold",color=_tc(cmap,M[i,j],vmin,vmax))
        ax.set_xticks(range(len(BENCHES))); ax.set_xticklabels([BLAB[b] for b in BENCHES],fontsize=8.5,rotation=35,ha="right")
        ax.set_yticks(range(len(LAB))); ax.set_yticklabels(LABT,fontsize=9); T(ax,tag,title)
        ax.set_xlim(-.5,len(BENCHES)-.5); ax.set_ylim(len(LAB)-.5,-.5)
    panel(axs[0],"a","H1  extinction  (ΔCR)",CM_LOSS,0,.22,
          lambda c:c.get("H1",{}).get("dCR"), lambda c:c.get("H1",{}).get("dCR_ci",[0,0])[0]>0, .13)
    panel(axs[1],"b","H2  provenance → correct  (AUC)",CM_GOOD,.5,.95,
          lambda c:c.get("H2",{}).get("n_independent",{}).get("dir_auc"),
          lambda c:c.get("H2",{}).get("n_independent",{}).get("dir_ci",[0,0])[0]>0.5, .78, fmt="{:.2f}")
    panel(axs[2],"c","selector ΔAcc  (logistic)",CM_DIV,-.16,.16,
          lambda c:c.get("H3",{}).get("prov_logistic_xfit",{}).get("delta"),
          lambda c:c.get("H3",{}).get("prov_logistic_xfit",{}).get("ci",[0,0])[0]>0, 0)
    fig.suptitle("Scale extension: the mechanism persists at a 24B open-weight anchor and a frontier API "
                 "(∗ = CI-significant; hatched cell = fails pre-registered validity gate)",
                 x=.012,ha="left",fontsize=11.5,fontweight="bold")
    save(fig,"fig13_scale_ext")

for name,fn in [("fig2",fig2),("fig3",fig3),("fig4",fig4),("fig5",fig5),("fig6",fig6),("fig7",fig7),("fig8",fig8),("fig9",fig9),("fig10",fig10),("fig10b",fig10b),("fig11",fig11),("fig12",fig12),("fig13",fig13)]:
    try: fn()
    except Exception as e:
        import traceback; print(f"!! {name} FAILED: {e}"); traceback.print_exc()
print("done.")
