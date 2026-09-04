# Not All Consensus Is Equal — Epistemic Provenance of Agreement in Multi-Agent LLM Debate

Journal extension of the conference IA-MAD paper (working manuscript: `paper/paper.tex`). Three peer LLMs
(`careful` qwen2.5:3b, `skeptic` phi4-mini, `bigpicture` llama3.2:3b) debate for ≤3 rounds over GSM8K /
StrategyQA / CommonsenseQA / BBH / TruthfulQA (5 seeds × 100 q), plus AQuA / ARC / MMLU for breadth and
alternative ensembles (7B, mixed-9B, 14B anchor, N=5, a 24B anchor, and a frontier API model) for
generalization. Every analysis is a replay of frozen transcripts in `data/debate_logs/`; GPU / API steps
are marked.

> **This document is the detailed engineering log.** The short project intro is `README.md`; the
> submission narrative is `paper/paper.tex`; the running research memory is in the maintainer's notes.

---

## TL;DR

Multi-agent debate (MAD) is supposed to make answers more reliable. We show it routinely **destroys**
correct answers: on 4 of 5 benchmarks a correct hypothesis that existed during the debate is gone by the
final vote, and simply trusting the pre-debate **independent** answers beats the debated majority. The
obvious fix — detect failing debates and intervene — hits a wall: *when* a debate fails is predictable
(hazard ROC 0.73–0.76) with a large **oracle** ceiling (+16pp), but *which* debate benefits is **not
identifiable** from the pre-intervention state (five pre-registered rescues fail). So we stop targeting
and fix the loss structurally: **preserve every hypothesis** and read its **provenance**.

The central scientific finding: **consensus is not epistemically homogeneous.** At matched total support,
an *independently generated* supporter carries far more correctness signal than a *socially adopted* one
(per-origin log-odds 0.92 vs −0.04; Δβ = 0.96). A compute-matched **self-consistency** control confirms
the mechanism from the other side: at equal inference budget, keeping samples independent (0.610) beats
letting them become socially coupled through debate (0.530).

**Contribution hierarchy (how to read the paper):**

```
   Scientific finding            Not all consensus is equal — origin beyond magnitude (H1/H2/H3)
        >  Diagnostic framework  Preserve every hypothesis + track its provenance (no extra LLM calls)
             >  Application       A provenance-aware selector (+3.5pp; a proof-of-usefulness, NOT a
                                  deployment method competitive with self-consistency)
```

**Method:** `Preserve → Track Provenance → Select` — no extra LLM calls, no external judge.

---

## The pipeline: research questions & the three hypotheses

```
RQ1  Does debate FORGET correct answers?                                          → YES  (H1)
     A correct hypothesis appears, then goes extinct through interaction. 11.6% of accuracy is
     online-recoverable; correct-but-forgotten in 12.5pp of debates; debate is NET-DESTRUCTIVE
     (round-0 independent > debated final on 4/5).      (e1_cascade_headroom, e1e4a_clean, e4a_*)
        ▼
RQ2  Is INFLUENCE the right lever to recover them?                                → NO (bridge)
     Influence tracks stubbornness (r=0.69), not competence (r=0.15); an influence-weighted vote
     LOSES to plain majority (0.502 < 0.528).                     (e2_mechanism_influence, voting)
        ▼
RQ3  Can we PREDICT which debates to fix and ACT on it?                           → predict≠act
     Failure IS predictable (hazard ROC 0.757) with a +16.5pp ORACLE ceiling, yet intervention
     BENEFIT is unidentifiable — 5 pre-registered rescues fail (cheap/semantic/probe/dose/traj).
        │  (e3_router, e5_counterfactual, e13_benefit_aware, e11_*, e14–e17)  [DEMOTED to a bridge]
        ▼
RQ4  Can PRESERVING hypotheses prevent the irreversible loss?                     → YES  (H1)
     Never destructively overwrite (H_t ⊆ H_{t+1}); select from the full history. CR 0.636→0.760
     (ΔCR +12.5pp, significant on all 5), pool ≈ 3 candidates.               (e18_hypothesis_preservation)
        ▼
RQ5  Is a hypothesis's PROVENANCE predictive of correctness?                      → YES  (H2)
     3 independent origins → 83% correct; 1 → 17%; emerges only in debate → 7%.
     n_independent dir-AUC 0.82, persistence 0.81.                                    (e22_provenance)
        ▼
RQ6  Beyond MAGNITUDE — does ORIGIN matter at matched support?                    → YES  (H3, HEADLINE)
     Controlling total support (they are r=0.83 collinear), an extra INDEPENDENT origin still
     predicts correctness (β_ind=0.589 [.512,.671]); decomposition β_ind 0.92 vs β_adopt −0.04
     (Δβ=0.96); survives within-question conditional logit (0.47 [.42,.53]).      (e26_independence)
        ▼
RQ7  Does provenance-aware selection improve decisions at NO extra cost?          → YES, conditional
     0.528→0.564 (+3.5pp), beats deployed majority & influence-vote; ~15% of the +23pp ceiling;
     regime-bound. Self-consistency control confirms "trust independent" direction.
        │  (e19_selector_bakeoff, e23_provenance_selector, e24_convergence_selector, e31_self_consistency)
        ▼
RQ8  Does it GENERALIZE across model families / scale / N?                        → broadly (see below)
     40-cell grid: H1 36/40 (40/40 positive), H2 38/40, selector 13/40. 3B→14B→24B open-weight
     + a frontier API model (deepseek-v4-flash).      (e25_*, e32_multiplicity)
```

---

## Key numbers (frozen, robust re-scoring; pooled over the core benchmarks unless noted)

Numbers use the robust extractor (`src/utils/answer_extract.py`): recovers `\boxed{}` answers hidden by
512-token truncation and **censors** genuinely-truncated ones.

### H1 — Debate forgets correct answers; preservation ceiling (`e18_hypothesis_preservation.json`)

| Benchmark | Maj. acc | CR_final | CR_history | **ΔCR [95% CI]** | Pool |
|---|--:|--:|--:|:--|--:|
| GSM8K | 0.732 | 0.824 | 0.920 | +0.096 [.070,.128] | 2.40 |
| BBH | 0.456 | 0.602 | 0.798 | **+0.196** [.156,.236] | 2.84 |
| StrategyQA | 0.600 | 0.750 | 0.908 | +0.158 [.112,.206] | 1.64 |
| CommonsenseQA | 0.638 | 0.806 | 0.936 | +0.130 [.100,.162] | 2.26 |
| TruthfulQA | 0.120 | 0.196 | 0.240 | +0.044 [.022,.068] | 6.31 |
| **Pooled** | **0.509** | **0.636** | **0.760** | **+0.125 [.108,.142]** | 3.09 |

Total headroom over the deployed majority ≈ **+25pp** (12.5 structural preservation + 12.7 selection).
Round-0 **independent** majority (0.560) beats the debated final majority (0.528) — debate is net-destructive.

### H2 — Provenance predicts correctness (`e22_provenance.json`, candidate-level)

| Signal | dir. AUC [95% CI] | monotone | P(correct \| value) |
|---|--:|:--|:--|
| n_independent (round-0 support) | 0.823 [.800,.845] | 4/5 | 0:.07 · 1:.17 · 2:.58 · 3:.83 |
| n_persist (rounds alive) | 0.806 [.785,.824] | 5/5 | 1:.06 · 2:.30 · 3:.57 |
| n_support (total votes) | 0.857 [.836,.877] | 2/5 | .05 → .84 |
| first round (late emergence) | 0.695 [.681,.709] | 4/5 | r0:.37 · r1:.08 · r2:.04 |

### H3 — Origin beyond magnitude (HEADLINE, `e26_independence.json`, 27,139 candidates, 5 ens × 8 bench)

Independent and total support are **collinear (r=0.83)**, so we compare at **matched total support**:

| n_sup | n_ind=0 | n_ind=1 | n_ind=2 | n_ind=3 |
|---|--:|--:|--:|--:|
| 4 | 0.21 [.15,.28] (156) | 0.44 [.40,.47] (739) | 0.66 [.62,.70] (458) | 0.74 [.60,.85] (43) |

At fixed support, correctness rises monotonically with the number of independent origins (Wilson CIs;
the overall rise is significant — n_ind=0/1/2 intervals non-overlapping — only the n=43 top cell is noisy).

- **Partial effect ctrl. total support:** β_ind = **0.589** [.512, .671]; +FE 0.590.
- **True decomposition:** per-origin log-odds β_ind **0.92** vs genuine peer-adoption β_adopt **−0.04**
  (n.s.) → **Δβ = 0.96** [.88, 1.05]. Persistence positive (0.83); adoption null after controlling it.
- **Not question difficulty:** within-question conditional logit β_ind = **0.474** [.419, .528]; within
  the same question the correct hypothesis carries more independent origins in 75% of pairs (57% at
  matched support).
- **Robust:** survives final-round-vote matching (1.03), conservative (bench,question) clustering (0.589),
  exposure-time adjustment (0.531). Significant on **7/8 benchmarks** (StrategyQA the boundary) and **5/5
  ensembles** (incl. 14B anchor, N=5).
- **Why the selector is only modest:** incremental candidate-AUC of n_ind over n_support is just **+0.007**
  (redundancy, VIF 3.2, ridge-stable) — a strong conditional signal need not give a large *ranking* gain.
- **E27 (`e27_surface_genuine.json`):** splitting adoption into surface vs genuine, the genuine−surface
  contrast **vanishes after controlling support** (−0.014 [−.111,.080]) → what matters is *origin*, not the
  reasoning-quality of social adoption. Appendix-only.

### RQ7 — Provenance-aware selection (`e23_provenance_selector.json`, 5 bench × 5 seed, gold-free)

| Selector | Pooled acc | Δ vs deployed majority [95% CI] | Extra LLM calls |
|---|--:|:--|--:|
| Final majority (deployed) | 0.528 | — | 0 |
| Conference influence-voting | 0.502 | −0.026 [−.036,−.018] | 0 |
| Round-0 (independent) majority | 0.560 | +0.032 [.015,.048] | 0 |
| **Provenance selector (ours)** | **0.564** | **+0.035 [.022,.049]** | **0** |
| Oracle (history-pool ceiling) | 0.760 | +0.232 | — |

Captures ~15% of the +23pp ceiling; **numerically close to** round-0-majority (0.564 vs 0.560; we report
**no direct paired test** of that small gap). Positioned as a downstream *application*, not a deployment
method competitive with self-consistency.

### Self-consistency control (`e31_self_consistency.json`, single-seed 500-q subset)

Reviewer question — is debate just inefficient sampling? Each E_orig backbone sampled N∈{3,5,9}× at T=0.7,
majority-voted; **SC-mixed** = N/3 samples from each backbone. **Budget matched by call count, not decoding
temperature.**

| Rule (N=9 ≈ debate's ~8.7 calls) | Pooled | Δ vs debate [95% CI], McNemar |
|---|--:|:--|
| Debate final majority | 0.530 | — |
| Round-0 majority | 0.542 | — |
| Best single-model SC (llama3.2:3b) | 0.596 | **+0.066** [.024,.108], p<0.01 |
| **SC-mixed (heterogeneous)** | **0.610** | **+0.080** [.042,.118], p<1e-4 |

At matched budget, independent sampling beats destructive debate (5/5 benches for the two stronger single
models; 4/5 for mixed, the exception being StrategyQA — the one net-constructive regime). This **confirms**
the thesis; it is not a baseline that defeats our method. Budget ladder N=3/5/9 in `tab:sc_budget`.

### Multiplicity — Benjamini–Hochberg FDR (`e32_multiplicity.json`, 40-cell grid, q=0.05)

| Family | raw sig /40 | **BH-FDR /40** |
|---|--:|--:|
| H1 extinction (ΔCR>0) | 35 | **35** (unchanged) |
| H2 provenance→correctness (dir-AUC>0.5) | 39 | **39** (unchanged) |
| selector ΔAcc>0 (conditional utility) | 14 | **8** |

The two *scientific* families are untouched by multiplicity correction; only the *conditional-utility*
family shrinks — exactly the paper's position. (Paper's "14/40" = either frozen variant; primary cross-fit
logistic alone = 13/40, 9/40 after BH-FDR.) Primary inference is the pooled/global H1–H3 tests; the 40-cell grid is
generalization/heterogeneity.

---

## Generalization grid (`e25_*`, frozen E23 method — never tuned)

Alt-ensemble logs go to `data/debate_logs/gen/<ensemble>/`; frozen originals are never touched. Generation
is GPU-/API-pinned; the verdict is zero-GPU and N-agnostic.

| Ensemble | Backbones (careful / skeptic / bigpicture) | Role | Status |
|---|---|---|---|
| `E_orig` (reference) | qwen2.5:3b / phi4-mini / llama3.2:3b | small (frozen logs) | done |
| `E_7b` | qwen2.5:7b / mistral:7b / llama3.1:8b | mid, cross-family | done |
| `E_mixed` | gemma2:9b / phi4-mini / llama3.2:3b | mixed capability | done |
| `E_strong` | qwen2.5:14b / phi4-mini / llama3.2:3b | 14B anchor (13.5 GB) | done |
| `E_n5` | + mistral:7b + qwen2.5:7b (N=5) | agent-count stress | done |
| **`E_strong24`** | **mistral-small3.1:24b** / phi4-mini / llama3.2:3b | **24B anchor** | **done** (H1 7/8, H2 8/8, selector 5/7 valid) |
| **`E_frontier`** | **deepseek-v4-flash ×3 personas** (cloud API) | **frontier-scale** | **done** (H1 8/8, H2 7/8; only 2/8 cells pass validity gate — outputs often unparseable) |

Benchmarks: 5 core + AQuA / ARC / MMLU (MATH was collected then **removed** by decision). Sign summary over
the 40-cell grid: **H1 36/40 (40/40 positive), H2 38/40, selector 13/40** (regime-bound, no significant
loss). Extension axes reported separately (`e25_scale_ext.json`, Fig. 13): E_strong24 (24B), E_frontier (deepseek). Applicability conditions (predictive, pre-registered in `paper/E25_newbench_prereg.md`): reranking
helps ⟺ **identifiability** (provenance informative; violated by binary StrategyQA) **∧ availability**
(a correct hypothesis is in the pool; TruthfulQA tops at ~24–31%). Analysis: `paper/E25_H3_generalization_analysis.md`.

---

## 2026-09 reviewer-response work & extensions

A tough ICML/NeurIPS-style meta-review drove the following. **Status tracker (2026-09-02):**

| Reviewer point | Status | Where / how |
|---|---|---|
| **#1** Scale too small / no frontier | ✅ done | E_strong24 (24B: H1 7/8, H2 8/8) + E_frontier (deepseek-v4-flash: H1 8/8, H2 7/8, only 2/8 cells inside validity gate) → range **3B→24B + frontier-scale API**; proprietary-frontier (GPT-4/Claude/Gemini) still open |
| **#2** E_strong(14B)/E_n5 only 1 seed | ✅ done | `scripts/e25_seed_topup_gen.py` generated seeds 43-44; E_strong/E_n5 now **3 seeds** on {gsm8k, commonsenseqa, bbh, strategyqa}; new 24B/frontier 3-seed on core |
| **#3** Contribution-vs-recommendation tension | ✅ done | diagnostic reframe (item 6) |
| **#4** "independent = self-consistency renamed" | ✅ done | novelty lock in Related Work + Intro (item 6); E_frontier homogeneous adds an empirical control |
| **#5** No multiplicity correction | ✅ done | BH-FDR E32 (item 5) |
| minor — matched-support cell n=43 no CI | ✅ done | Wilson CIs, 0.74 [0.60,0.85] (item 4) |
| minor — StrategyQA post-hoc claim | ✅ done | already hedged: "a boundary rather than a demonstrated cause" |
| minor — persona⊥backbone confound | ✅ acknowledged | Limitations (future work); E_frontier (same backbone, persona-only diversity) = partial control |
| minor — no confidence / ReConcile baseline | ✅ done | Limitations statement: frequency-confidence ≈ round-0/support baselines; calibrated/self-reported confidence + ReConcile-style aggregators = future work (backend exposes no logprobs) |
| minor — prose too number-dense | 🟡 substantial | addressed by the 20-25% trim (item 1); further micro-pass optional |

Detail:

1. **Main-text trim & restructure** — 6 main figures (`figure*`, full width), rest → appendix, every
   figure/table cited (`Figure X` / `Table X`, no "main vs appendix" wording). Discussion 5→3 subsections;
   selector 6→1 paragraphs; generalization 4→1; removed a redundant subsection. Main body ≈ 9.5k words.
2. **Correctness fixes** (all verified against source): `n_inh` cumulative-vs-peer-conditioned bug → true
   peer-conditioned adoption decomposition (`e28_h3_robust.py`); unified baseline 0.528; AQuA N=1302 (100 q,
   E_orig 20-pilot subset); algorithm pseudocode O(NT) precompute; consistent cross-references.
3. **Self-consistency baseline (E31)** — `experiments/e31_self_consistency.py` (+`--ci` cache-only paired
   bootstrap/McNemar → merged into `e31_self_consistency.json`). Integrated at 4 spots (Intro, §5.7,
   appendix `tab:selfconsistency` + budget ladder `tab:sc_budget`, Limitations). Framed as **confirmatory**.
4. **Matched-support Wilson CIs** — added to `tab:matched` prose + caption (flagged n=43 cell → [0.60,0.85]).
5. **BH-FDR multiplicity (E32)** — `experiments/e32_multiplicity.py`; fixed a data gap first
   (`e25_generalization.json` had 9 cells not 15 — the "bare `--verdict` = 3 pilot benches" bug; regenerated,
   9 originals byte-identical). Result: H1/H2 unchanged, selector 14→8. Integrated in §5.1 + appendix.
6. **Diagnostic reframe** — locked the hierarchy *finding > framework > selector*; locked the
   novelty-vs-self-consistency distinction in Related Work ("SC aggregates independent samples; we ask
   whether, once agents interact, equal observed consensus retains equal evidential value — it does not");
   stated the selector does not compete with SC as a deployment method.
7. **Larger open-weight anchor (E_strong24, 24B)** and **frontier API model (E_frontier,
   deepseek-v4-flash)** — see the grid above; both **launched 2026-09-02**, logs under `results/logs/`.
   To wire the cloud model, an **OpenAI-compatible backend** was added to the framework (was Ollama-only):
   `src/agents/base_agent.py::_call_openai_chat`, threaded through `run_debate._build_agents`
   (`backend` / `api_key_env`); default stays `ollama` so nothing else changed.
8. **Seed top-up (#2)** — `scripts/e25_seed_topup_gen.py`: waits for E_strong24 to finish, then generates
   seeds 43-44 for E_strong (14B) + E_n5 on {gsm8k, commonsenseqa, bbh, strategyqa} so those exact cells get
   3 seeds. Launched (queued) 2026-09-02, gpu0-pinned, resumable.
9. **Confidence / ReConcile baseline statement** — added to Limitations (cites `chen2024reconcile`), closing
   the "weak baselines" minor point honestly (frequency-confidence already subsumed by round-0/support;
   calibrated/self-reported confidence unavailable from the backend).

**Live runs (2026-09-02):**

| Run | Resource | Purpose | Status |
|---|---|---|---|
| `e25_strong24_gen.py` | gpu0 `:11436` (Mistral-24B, ~89 s/debate) | #1 larger anchor | ✅ done, 16 cells (verdict `e25_scale_ext.json`) |
| `e25_frontier_gen.py` | cloud API (deepseek-v4-flash) | #1 frontier-scale | ✅ done, 16 cells (billing-402 resume used `deepseek-chat` = same underlying v4-flash) |
| `e25_seed_topup_gen.py` | gpu0/gpu1 ollama | #2 seed variance | ✅ done (seeds 43-44 on core) |

**Done 2026-09-04:** final numerical audit + paper claim updates — all verdict tables, `fig_cells`, e26/e28/e30/e32
and figures regenerated over the complete logs (27,139 candidates); paper.tex + README numbers updated (range
"3B–14B"→"3B–24B + frontier-scale API"); E_strong24 / E_frontier wired into `e25_scale_ext.json` + Fig. 13.

**Still open (by decision):** (#6) release scope of derived candidate/provenance tables (awaiting sign-off);
proprietary-frontier validation (#1 residual — deepseek-v4-flash is frontier-*scale API*, not GPT-4/Claude/Gemini).

---

## Conference → journal bridge

The IA-MAD influence analysis still runs and is now a **baseline layer** (RQ2): influence tracks stubbornness
(r=0.69) not competence (r=0.15); rank is stable across benchmarks (Spearman 0.80); but the influence-aware
**vote loses to plain majority** (0.502 vs 0.528). The story moves from *influence* to *preservation &
provenance*. (`experiments/analyze_all.py`.)

---

## Alternative approaches tested (supplementary, not in the main narrative)

Ruled out en route to the method; each is a script + result table kept for completeness: semantic mechanism
router (`e3b_*`); hazard-guided **controller** + robustness battery (`e6`–`e9`, `e12`) — matches always-on at
~half the interventions but its edge over a random budget is ≈0, which is *why* RQ3 concludes targeting is
hard; four benefit-prediction rescues (`e14` semantic, `e15` active probe, `e16` dose-response, `e17`
trajectory) — all fail the same pre-registered gate; deterministic arithmetic verifier (`e20`); external LLM
judge over the pool (`e21`) — a capable judge does better *solving* than *selecting* from a weak pool.

---

## Setup

3 peer LLMs debate ≤3 rounds; each sees the others' answers and may revise. Local inference via **Ollama**;
frontier via an **OpenAI-compatible cloud API** (DeepSeek).

| Role (persona) | Base model (E_orig) | Temp |
|---|---|--:|
| `careful` | `qwen2.5:3b-instruct` | 0.3 |
| `skeptic` | `phi4-mini:latest` | 0.5 |
| `bigpicture` | `llama3.2:3b` | 0.5 |

Benchmarks: GSM8K, StrategyQA, CommonsenseQA, BBH, TruthfulQA (core) + AQuA, ARC, MMLU (breadth). Seeds
`[42–46]`. See `config/config.yaml`.

---

## How to run

**Core pipeline — all zero-GPU** (replays `data/debate_logs/*.jsonl`):

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES="" python experiments/e1e4a_clean.py            # H1 recovery headroom
CUDA_VISIBLE_DEVICES="" python experiments/e4a_extinction_anatomy.py # extinction mechanisms
CUDA_VISIBLE_DEVICES="" python experiments/e2_mechanism_influence.py # influence can't diagnose (AUC 0.49)
CUDA_VISIBLE_DEVICES="" python experiments/e3_router.py              # hazard detector (ROC 0.757)
CUDA_VISIBLE_DEVICES="" python experiments/e13_benefit_aware.py      # failure != benefit (oracle +16.5pp)
CUDA_VISIBLE_DEVICES="" python experiments/e18_hypothesis_preservation.py  # CR 0.636 -> 0.760
CUDA_VISIBLE_DEVICES="" python experiments/e22_provenance.py         # provenance predicts correctness (H2)
CUDA_VISIBLE_DEVICES="" python experiments/e26_independence.py       # origin beyond magnitude (H3 headline)
CUDA_VISIBLE_DEVICES="" python experiments/e28_h3_robust.py          # true adoption decomposition + robustness
CUDA_VISIBLE_DEVICES="" python experiments/e27_surface_genuine.py    # surface vs genuine adoption (appendix)
CUDA_VISIBLE_DEVICES="" python experiments/e23_provenance_selector.py  # the selector (+3.5pp)
CUDA_VISIBLE_DEVICES="" python experiments/e30_reviewer_robustness.py  # VIF/ridge/error-corr/counts
CUDA_VISIBLE_DEVICES="" python experiments/e31_self_consistency.py --ci  # SC paired CIs (needs cache; see below)
CUDA_VISIBLE_DEVICES="" python experiments/e32_multiplicity.py       # BH-FDR over the 40-cell grid
```

**GPU / API generation** (never run analysis on the GPU):

```bash
# 24B larger anchor — gpu0-pinned Ollama :11436, detached, resumable
CUDA_VISIBLE_DEVICES="" setsid nohup python scripts/e25_strong24_gen.py > results/logs/e25_strong24.log 2>&1 &

# Frontier via cloud API — PURE API, zero GPU. Key from a file OUTSIDE the repo.
export DEEPSEEK_API_KEY="$(cat ~/.deepseek_key)"   # never commit; read at launch
export DEEPSEEK_MODEL=deepseek-v4-flash
python scripts/e25_frontier_gen.py --smoke         # 2-debate sanity check first
CUDA_VISIBLE_DEVICES="" setsid nohup python scripts/e25_frontier_gen.py > results/logs/e25_frontier.log 2>&1 &

# self-consistency generation (gpu0 :11436) then paired CIs
CUDA_VISIBLE_DEVICES="" python experiments/e31_self_consistency.py   # generate + eval (resumable cache)

# generalization verdict (zero-GPU) once logs exist
CUDA_VISIBLE_DEVICES="" python -c "import experiments.e25_generalization as e; \
  e.verdict(ensembles=['E_orig','E_7b','E_mixed'], benches=['gsm8k','strategyqa','commonsenseqa','bbh','truthfulqa'])"
CUDA_VISIBLE_DEVICES="" python experiments/e25_newbench_verdict.py   # breadth + scale disposition
```

---

## Repository layout

```
influence-mad/
├── config/config.yaml                # agents (persona↔model), benchmarks, seeds, debate params
├── data/debate_logs/                 # {bench}_seed{42..46}.jsonl  ← frozen transcripts (input to ALL analysis)
│   └── gen/<ensemble>/               # alt-ensemble logs (E_7b, E_mixed, E_strong, E_n5, E_strong24, E_frontier)
├── src/
│   ├── agents/base_agent.py          # Ollama /api/generate + OpenAI-compatible /chat/completions backends
│   ├── agents/peer_agent.py          # personas + INIT/REFINE prompts
│   ├── influence_estimation/ voting/ # conference layer (PageRank, influence-aware vote) = baseline
│   └── utils/{scoring,answer_extract}.py   # is_correct + robust_answer (recover \boxed{}, censor truncation)
├── experiments/
│   ├── e1*/e4a*                       # H1 extinction & headroom
│   ├── e2*/e3*/e5*/e11*/e13*          # RQ2/RQ3 influence + predict≠act (bridge)
│   ├── e18*                           # H1 preservation ceiling
│   ├── e22*/e26*/e27*/e28*           # H2 provenance + H3 origin-beyond-magnitude + robustness
│   ├── e19*/e23*/e24*/e31*           # selector + self-consistency control
│   ├── e25_generalization.py e25_newbench_verdict.py e25_scale_verdict.py e32_multiplicity.py  # grid + FDR
│   └── … (supplementary: e3b_*, e6–e9, e12, e14–e17, e20, e21)
├── scripts/                          # e25_gpu0_gen.py, e25_newbench_gen.py, e25_strong24_gen.py, e25_frontier_gen.py
├── results/{tables,figures,logs}/
├── paper/                            # paper.tex (manuscript), references.bib, E25_*_prereg.md, figures
└── requirements.txt
```

---

## Honest caveats (kept in the paper)

- **The selector is modest** (+3.5pp, ~15% of the ceiling) and is only *numerically close* to
  round-0-majority (no paired test of the small gap) — independent agreement is what wins, not the score's
  sophistication. ~85% of the ceiling is the standing open problem. It is a *proof-of-usefulness*, not a
  deployment method that beats self-consistency.
- **H3 is observational** (a logistic association, not a manipulation): survives a within-question control
  but residual candidate-level confounding is possible; "independent" = interaction-independent, not
  statistically independent (round-0 errors are correlated 1.5–158× over the independence prediction).
- **Gains are regime-dependent**: debate is net-destructive on 4/5 benches; StrategyQA is the constructive
  counterexample (H3 boundary); TruthfulQA fails availability.
- **All selection numbers are offline replays** of a frozen destructive-protocol debate; a live
  non-destructive protocol (HP-MAD) with a matched-class / self-consistency verifier is future work.
- **Scale, not proprietary-frontier**: the range is 3B → 24B open-weight + deepseek-v4-flash (a
  frontier-scale *API* model). No GPT-4/Claude/Gemini data point yet.
- **Self-consistency at matched budget beats destructive debate** — reported as confirmatory evidence for
  the thesis, and as the honest practical recommendation (answer independently rather than debate).

---

> **GPU protocol.** Debate generation runs through a **dedicated GPU-pinned** Ollama instance, one per card,
> to avoid disturbing the other user's job: `CUDA_VISIBLE_DEVICES=0 OLLAMA_VULKAN=0
> OLLAMA_HOST=127.0.0.1:11436 ollama serve` pins **gpu0**; the mirror on `:11435` / `CUDA_VISIBLE_DEVICES=1`
> pins **gpu1**. Pinning the client is **not** enough — the system server on `:11434` is unpinned, and
> `OLLAMA_VULKAN=0` is required or Ollama re-enumerates the other card via Vulkan. Always launch detached
> (`setsid nohup … < /dev/null &`) so a closing session can't kill it, and **yield whichever card the other
> user is on — never preempt.** Frontier runs use a cloud **API** (no GPU). Cloud API keys live **outside the
> repo** (e.g. `~/.deepseek_key`, chmod 600), read at launch via `$(cat …)` — never committed, never echoed.
> Zero-GPU analyses run under `CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python`.
