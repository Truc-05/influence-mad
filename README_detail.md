# FailSafe-MAD — Predicting Failure Is Not Predicting Benefit: Preservation & Provenance in Multi-Agent LLM Debate

Journal extension of the conference IA-MAD paper. Three peer LLMs (`careful` qwen2.5:3b, `skeptic`
phi4-mini, `bigpicture` llama3.2:3b) debate for ≤3 rounds over GSM8K / StrategyQA / CommonsenseQA /
BBH / TruthfulQA, 5 seeds × 100 questions. Everything below is a replay of the frozen transcripts in
`data/debate_logs/`; the two GPU steps are marked.

---

## TL;DR

Multi-agent debate (MAD) is supposed to make answers more reliable. We show it routinely **destroys**
correct answers: on 4 of 5 benchmarks a correct hypothesis that existed during the debate is gone by
the final vote, and simply trusting the pre-debate **independent** answers beats the debated majority
(56.0% vs 52.8% pooled). The obvious fix — detect failing debates and intervene — runs into a wall:
*when* a debate fails is predictable (hazard ROC 0.73) and intervention has a large **oracle** ceiling
(+16pp), but *which* debate benefits is **not identifiable** from the pre-intervention state. So we
stop targeting and fix the loss structurally: **preserve every hypothesis** (never let interaction
erase one) and **re-select by provenance**. Preservation lifts correct-answer recall 63.6% → 76.0%,
and a zero-extra-cost provenance selector beats the deployed majority by **+3.5pp** (and beats the
conference influence-vote, which itself loses to majority).

**Method:** `Preserve → Track Provenance → Select` — no extra LLM calls, no external judge.

---

## The pipeline: five research questions

```
RQ1  Does debate FORGET correct answers?                                         → YES
     A correct hypothesis appears, then goes extinct through social interaction.
     11.6% of accuracy is online-recoverable; the correct answer is present-but-forgotten in
     12.5pp of debates; debate is NET-DESTRUCTIVE (round-0 independent > debated final on 4/5).
        │  (e1_cascade_headroom, e1e4a_clean, e4a_extinction_anatomy)
        ▼
RQ2  Can influence / failure prediction tell us WHAT ACTION HELPS?               → NO
     Global influence can't diagnose the mechanism (AUC 0.49); failure IS predictable
     (hazard ROC 0.73–0.76) and intervention CAN rescue (+16pp preservation, E5); yet
     intervention BENEFIT is unpredictable (AUC 0.58, targeting ≈ random) despite a +16pp
     oracle ceiling. StrategyQA is the clean proof: failure ROC 0.87 but headroom +0.8pp.
     ⇒  PREDICTABILITY ≠ ACTIONABILITY.
        │  (e2_mechanism_influence, e3_router, e5_counterfactual, e11_*, e13_benefit_aware)
        ▼
RQ3  Can PRESERVING hypotheses prevent the irreversible loss?                    → YES
     Never destructively overwrite a hypothesis (H_t ⊆ H_{t+1}); select from the full history.
     Candidate recall 0.636 → 0.760 (ΔCR +12.5pp, significant on all 5 benchmarks), pool ≈ 3.
        │  (e18_hypothesis_preservation)
        ▼
RQ4  Is a hypothesis's PROVENANCE predictive of its correctness?                 → YES, strongly
     Produced independently by 3 agents → 83% correct; by 1 → 17%; only emerges in debate → 7%.
     n_independent AUC 0.82, persistence 0.81, first-round 0.70.
        │  (e22_provenance)
        ▼
RQ5  Can PROVENANCE-AWARE selection improve decisions with NO extra inference?   → YES
     Preserve → Track Provenance → Select: 52.8% → 56.4% (+3.5pp, CI>0, 4/5 benches, 0 extra calls),
     beating the deployed majority AND the conference influence-vote (50.2%). It ties the strong
     round-0 baseline — because debate is mostly destructive — which we report and explain.
        │  (e19_selector_bakeoff, e23_provenance_selector, e24_convergence_selector)
        ▼
RQ6  Does it GENERALIZE across model families / N-agents?                        → PILOT RUNNING (E25)
     Freeze the E23 method; re-generate debates for alternative ensembles and re-run the
     EXACT frozen verdict. Pilot: 3 ensembles × {gsm8k, strategyqa, csqa} × N=3.
        │  (e25_generalization)
```

**Five results the reader should see:** `forget (RQ1) → predict≠act (RQ2) → preserve (RQ3) →
provenance (RQ4) → select (RQ5)`.

---

## Key numbers (frozen, robust re-scoring; pooled over the 4 main benchmarks × 5 seeds unless noted)

Numbers use the robust extractor (`src/utils/answer_extract.py`): it recovers `\boxed{}` answers hidden
by 512-token truncation and **censors** genuinely-truncated ones so they don't inflate the headroom.

### RQ1 — Debate forgets correct answers

**Recovery headroom** (`e1e4a_clean.json`):

| Benchmark | Final majority | Online-recoverable | **Behavioral** (censored removed) |
|---|--:|--:|--:|
| GSM8K | 73.2% | 8.8% | 7.4% |
| StrategyQA | 60.0% | 11.0% | 11.0% |
| CommonsenseQA | 63.8% | 12.4% | 12.0% |
| BBH | 45.6% | 17.8% | 15.8% |
| **Pooled** | **60.7%** | **12.5%** | **11.6%** |

**Extinction anatomy** (`e4a_extinction_anatomy.json`): two behavioral mechanisms — social adoption and
self-error drift; the GSM8K open-answer anchor is self-error-led (self 57% ≥ social 43%). **73% of
adoptions are "surface"**: the answer copies a peer while the reasoning does *not* converge.

**Net-destructive** (5 benches, from RQ5's selector analysis): the pre-debate **round-0 independent
majority (56.0%) beats the debated final majority (52.8%)** — debate erases part of the wisdom of the
independent agents.

### RQ2 — Predictability ≠ actionability

**Influence can't diagnose the mechanism** (`e2_mechanism_influence.json`, `|A|`-controlled, grouped-CV):

| Feature set | AUC (social vs self) |
|---|--:|
| Influence-only (PageRank, stubbornness, Mᵢⱼ) | **0.49** ≈ chance |
| Structural-only (positive control) | 0.63 |

**Failure IS predictable** (`e3_router.json`, Stage-A hazard): AUPRC 0.24 (lift ~2×), ROC **0.757**;
LOBO across benchmarks 0.67–0.75 — generalizes, including open-answer. (Ranks well; no high-precision
operating point at 12% prevalence.)

**Intervention CAN rescue** (`e5_counterfactual.json`, first GPU run; re-query forward from the trigger
prefix, gold-free): the `combined` (self-verify + influence-resistant) arm lifts preservation **+12pp**
macro / **+16pp** pooled and final-majority recovery **+7pp** at ~equal call count (n=116).

**But benefit is unidentifiable** (`e13_benefit_aware.json`): both counterfactual arms are stored, so
ITE is observed. The **oracle** ceiling is large (**+16.5pp** pooled, reached at ~20% budget), but the
individual effect is unpredictable — Y0 predictable (AUC 0.73 = the hazard), Y1 predictable (0.67),
their **difference not** (effect AUC 0.58, sign anti-correlated); a learned CATE controller does **not**
beat random. **StrategyQA is the clean example** (`e11_*`): failure ROC **0.87** yet intervention
headroom is **+0.8pp** — detectability without actionability.

### RQ3 — Hypothesis preservation (`e18_hypothesis_preservation.json`)

Correct-answer recall at the final round vs across the whole history; ΔCR = forgotten-correct rate
(cluster-boot 95% CI over questions):

| Benchmark | Maj. acc | CR_final | CR_history | **ΔCR [95% CI]** | Pool |
|---|--:|--:|--:|:--|--:|
| GSM8K | 0.732 | 0.824 | 0.920 | +0.096 [.070,.128] | 2.40 |
| BBH | 0.456 | 0.602 | 0.798 | **+0.196** [.156,.236] | 2.84 |
| StrategyQA | 0.600 | 0.750 | 0.908 | +0.158 [.112,.206] | 1.64 |
| CommonsenseQA | 0.638 | 0.806 | 0.936 | +0.130 [.100,.162] | 2.26 |
| TruthfulQA | 0.120 | 0.196 | 0.240 | +0.044 [.022,.068] | 6.31 |
| **Pooled** | **0.509** | **0.636** | **0.760** | **+0.125 [.108,.142]** | 3.09 |

Significant on all 5 benchmarks; total headroom over the deployed majority is **+25pp**; the pool holds
only ~3 hypotheses, so the recovered answers are not noise. Largest on BBH — where targeting failed most.

### RQ4 — Provenance predicts correctness (`e22_provenance.json`, candidate-level)

| Signal | dir. AUC [95% CI] | monotone | P(correct \| value) |
|---|--:|:--|:--|
| n_independent (round-0 support) | 0.823 [.800,.845] | 4/5 | 0:.07 · 1:.17 · 2:.58 · 3:.83 |
| n_persist (rounds alive) | 0.806 [.785,.824] | 5/5 | 1:.06 · 2:.30 · 3:.57 |
| n_support (total votes) | 0.857 [.836,.877] | 2/5 | .05 → .84 |
| first round (late emergence) | 0.695 [.681,.709] | 4/5 | r0:.37 · r1:.08 · r2:.04 |

### RQ5 — Provenance-aware selection (`e23_provenance_selector.json`, 5 benches × 5 seeds, gold-free)

| Selector | Pooled acc | Δ vs deployed majority [95% CI] | Extra LLM calls |
|---|--:|:--|--:|
| Final majority (deployed) | 0.528 | — | 0 |
| Conference influence-voting | 0.502 | −0.026 [−.036,−.018] | 0 |
| Round-0 (independent) majority | 0.560 | +0.032 [.015,.048] | 0 |
| **Provenance selector (ours)** | **0.564** | **+0.035 [.022,.049]** | **0** |
| Oracle (history-pool ceiling) | 0.760 | +0.232 | — |

`S(h) = w₁·n_independent + w₂·n_persist + w₃·n_support − w₄·first_round` (equal-weight, or logistic
weights **cross-fit by question** so no weight sees its own test fold). Per-bench Δ vs majority: gsm8k
+0.048, csqa +0.084, bbh +0.036, tqa +0.020, strategyqa −0.012. It **ties** round-0-majority (both
0.560) — reported, not hidden — because debate is net-destructive; a convergence-gated variant
(`e24`) correctly recovers the one **constructive-debate** benchmark (StrategyQA, 0.580→0.604) but a
single global rule cannot win both regimes. This is why provenance (which *keeps* independent evidence
while *allowing* genuine revision) is the right framework, not a hard "always trust round-0" rule.

### RQ6 — Generalization of the frozen method

The E23 method is **frozen** — we do not tune it. We re-generate debates for alternative ensembles and
re-run the *exact* frozen verdict (`e25_generalization.py` reuses `e18/e22/e23` through a `logdir=` seam)
to test three pre-committed hypotheses, cell by cell:

- **H1** extinction generalizes — ΔCR > 0 (reuses E18)
- **H2** independent provenance predicts correctness — `n_independent` dir. AUC > chance (reuses E22)
- **H3** provenance selector still beats the deployed final-majority — ΔAcc CI-lo > 0 (reuses E23)

Alt-ensemble logs go to `data/debate_logs/gen/<ensemble>/` so the frozen originals are **never touched**;
generation is GPU-pinned, the verdict is zero-GPU and **N-agnostic** (validated on mock N=5 logs).

**Stage 1 — model family × 5 core benchmarks (DONE, `e25_generalization.json`).** Three N=3 ensembles
across gsm8k / strategyqa / csqa / bbh / truthfulqa, seeds 42–45 on the important cells:

| Ensemble | Backbones (careful / skeptic / bigpicture) | Role |
|---|---|---|
| `E_orig` (reference) | qwen2.5:3b / phi4-mini / llama3.2:3b | small (reuses frozen logs) |
| `E_7b` | qwen2.5:7b / mistral:7b / llama3.1:8b | mid, cross-family |
| `E_mixed` | gemma2:9b / phi4-mini / llama3.2:3b | mixed capability |

Result (sign test, 15 cells): **H1 15/15 · H2 14/15 · H3 7/15**. The two *scientific* claims generalize
across families including BBH (H2 fails only on `E_7b`/strategyqa — binary y/n → weak independent signal).
The selector's *accuracy* win (H3) is **regime-structured, not intermittent**: **7/9 on multi-answer
reasoning** (gsm8k/csqa/bbh) vs **0/6 on degenerate benches** (strategyqa/truthfulqa). Two applicability
conditions explain every cell — reranking helps ⟺ **identifiability** (provenance is informative;
StrategyQA's binary y/n is ≈chance) **∧ availability** (the correct hypothesis is in the pool;
TruthfulQA's ceiling tops at ~24–31%). No fail cell shows a *significant* loss (every CI includes 0).
Analysis: `paper/E25_H3_generalization_analysis.md`.

**Stage 2 — breadth + scale (in progress, pre-registered).** `paper/E25_newbench_prereg.md` (+ addendum)
locks the disposition of every new cell *before* results, as an out-of-sample test of the
identifiability∧availability framework — **not** a hunt for more H3 passes:

| Axis | Cells | GPU | Status |
|---|---|---|---|
| **+4 benchmarks** AQuA-RAT · ARC-Challenge · MMLU · MATH | `E_7b`/`E_mixed`/`E_orig3b` × 4 new benches, seed42 n=100 | gpu1 `:11435` | generating |
| **+strong-anchor** `E_strong` = qwen2.5:14b / phi4-mini / llama3.2:3b (parallels `E_mixed`, isolates the 14B anchor; 13.5 GB fits one card) | × 9 benches, seed42 n=100 | gpu0 `:11436` | generating |
| **+N=5** `E_n5` = qwen2.5:3b / phi4-mini / llama3.2:3b / mistral:7b / qwen2.5:7b (E_orig + 2 mid) | × 9 benches, seed42 n=100 | gpu0 `:11436` | generating |

Cells escalate to seeds 43–44 **only** if they pass the Step-A validity gate (parse ≤ 15% censored,
0.10 ≤ maj-acc ≤ 0.95, CR_history ≥ 0.15) **and** show H1∧H2 — an H3 near-miss is never a reason to
expand (anti-chasing). Pre-registered prediction: the four MC/sticky benchmarks are **H3-difficult** for
named reasons while H1/H2 hold where the regime is valid; a confirmed prediction turns the post-hoc §9
characterization into a predictive theory. Final ladder = **small → mid → mixed → strong** + **N=3 → N=5**,
≈7 checkpoints across 5 families (Qwen/Phi/Llama/Mistral/Gemma), 3B–14B. The controller (E6–E12) is
**not** re-generalized — it is dead (RQ2).

---

## Conference → journal bridge

The IA-MAD analysis still runs and is now a **baseline layer**: influence does not track competence
(r = 0.15) but tracks stubbornness (r = 0.69), and influence rank is stable across benchmarks
(Spearman 0.80). The journal shows its limit: influence **cannot** diagnose the failure mechanism (RQ2,
AUC 0.49) and the influence-aware **vote loses to plain majority** (RQ5, 50.2% vs 52.8%). The story
moves from *influence* to *preservation & provenance*. (`experiments/analyze_all.py`.)

---

## Alternative approaches tested (supplementary, not in the main narrative)

To reach the method above we ruled out several alternatives; each is a script + result table, kept for
completeness only: a semantic mechanism router (`e3b_*`); a hazard-guided adaptive **controller** and
its robustness battery (`e6`,`e7`,`e8`,`e9`,`e12`) — it matches always-on at ~half the interventions
but its detector edge over a random budget is ≈0, which is *why* RQ2 concludes targeting is the hard
part; four benefit-prediction rescues (`e14` semantic, `e15` active probe, `e16` dose-response, `e17`
trajectory) — all fail the same preregistered gate, confirming benefit is unidentifiable; a
deterministic arithmetic verifier (`e20`); and an external LLM judge over the pool (`e21`) — a capable
judge does better solving than selecting from a weak pool. None enters the main text.

---

## Setup

3 peer LLMs debate ≤3 rounds; each sees the others' answers and may revise. Everything is logged
(per-round answer, whether it changed). Local inference via **Ollama**.

| Role (persona) | Model |
|---|---|
| `careful` | `qwen2.5:3b-instruct` |
| `skeptic` | `phi4-mini:latest` |
| `bigpicture` | `llama3.2:3b` |

Benchmarks: **GSM8K, StrategyQA, CommonsenseQA, BBH, TruthfulQA** (TruthfulQA off the main axis —
open-ended, ~12% baseline). Seeds `[42–46]`, 100 questions/seed → 500/benchmark. See `config/config.yaml`.

---

## How to run

**Core pipeline — all zero-GPU** (replays `data/debate_logs/*.jsonl`), grouped by research question:

```bash
pip install -r requirements.txt

# RQ1 — debate forgets correct answers
python experiments/e1_cascade_headroom.py        # headroom decomposition (+ figure)
python experiments/e1e4a_clean.py                # frozen E1+E4a numbers under robust extraction
python experiments/e4a_extinction_anatomy.py     # extinction mechanisms (+ figure)

# RQ2 — predictability != actionability
python experiments/e2_mechanism_influence.py     # influence cannot diagnose the mechanism (AUC 0.49)
python experiments/e3_router.py                  # Stage-A hazard detector (ROC 0.757, LOBO)
python experiments/e13_benefit_aware.py          # failure != benefit: oracle +16.5pp, CATE ~ random
python experiments/e11_benchmark_expansion.py    # detectability landscape; StrategyQA ROC 0.87
#   E5 intervention is GPU (Ollama); eval reuses the stored arms — see below

# RQ3/RQ4/RQ5 — preserve, provenance, select
python experiments/e18_hypothesis_preservation.py  # preservation ceiling: CR 0.636 -> 0.760
python experiments/e22_provenance.py               # provenance predicts correctness
python experiments/e19_selector_bakeoff.py         # gold-free selector baselines (round-0 etc.)
python experiments/e23_provenance_selector.py      # the method: Preserve -> Provenance -> Select (+3.5pp)
python experiments/e24_convergence_selector.py     # round-0 analysis + constructive/destructive regimes
```

**GPU steps (Ollama, gpu1-pinned)** — data collection + the E5 counterfactual intervention:

```bash
ollama pull qwen2.5:3b-instruct && ollama pull phi4-mini && ollama pull llama3.2:3b
python experiments/run_pipeline.py --benchmark gsm8k --config config/config.yaml   # (re)collect transcripts
python experiments/e5_counterfactual.py --bench gsm8k,bbh --per-bench 40            # intervention (rescue)

# RQ6 — generalization of the FROZEN method (E25): GPU generation + zero-GPU verdict
python experiments/e25_generalization.py --selftest    # mock wiring check (no GPU)
python experiments/e25_generalization.py --gen-pilot   # gpu1-pinned: 3-ensemble × 5-bench debates (data/debate_logs/gen/)
python experiments/e25_generalization.py --verdict     # Stage-1 H1/H2/H3 across cells (zero-GPU) -> e25_generalization.json
# Stage 2 — pre-registered breadth + scale (paper/E25_newbench_prereg.md):
python scripts/e25_newbench_gen.py                     # gpu1 :11435 — +AQuA/MATH/MMLU/ARC on E_7b/E_mixed/E_orig3b
python scripts/e25_gpu0_gen.py                         # gpu0 :11436 — E_strong (14B) + E_n5 (N=5) × 9 benches
python experiments/e25_newbench_verdict.py             # zero-GPU verdict + §3 disposition (per-cell logdir)
```

Supplementary experiments (alternatives above) live under the same `experiments/e*` prefix and write to
`results/tables/`; they are not needed to reproduce the main pipeline.

---

## Repository layout

```
influence-mad/
├── config/config.yaml              # agents (persona↔model), benchmarks, seeds, debate params
├── data/debate_logs/               # {bench}_seed{42..46}.jsonl  ← raw transcripts (input to ALL analysis)
├── src/
│   ├── agents/                     # PeerAgent + personas, Ollama wrappers (data collection & E5)
│   ├── influence_estimation/       # influence matrix + PageRank            (conference; baseline layer)
│   ├── influence_voting/           # majority / influence-aware voting       (conference baseline)
│   └── utils/{scoring,answer_extract}.py   # is_correct + robust_answer (recover \boxed{}, censor truncation)
├── experiments/
│   │ # --- core pipeline (RQ1→RQ5) ---
│   ├── e1_cascade_headroom.py e1e4a_clean.py e4a_extinction_anatomy.py     # RQ1
│   ├── e2_mechanism_influence.py e3_router.py e5_counterfactual.py         # RQ2
│   ├── e11_benchmark_expansion.py e13_benefit_aware.py                     # RQ2
│   ├── e18_hypothesis_preservation.py                                      # RQ3
│   ├── e22_provenance.py                                                   # RQ4
│   ├── e19_selector_bakeoff.py e23_provenance_selector.py e24_convergence_selector.py  # RQ5
│   └── … (supplementary alternatives: e3b_*, e6–e9, e12, e14–e17, e20, e21)
├── results/{tables,figures,logs}/
├── paper/                          # PAPER_OUTLINE.md (C1–C4 spine), HPMAD_results.tex (RQ3–RQ5), *.tex, *.bib
└── requirements.txt
```

---

## Honest caveats (kept in the paper)

- **The provenance selector is modest** (+3.5pp, ~14% of the ceiling) and **ties round-0-majority** —
  independent agreement is what wins, not the score's sophistication. Framed as "provenance generalizes
  trust-early to the constructive-debate case at equal cost." ~86% of the ceiling is the open problem.
- **Gains are benchmark-regime-dependent**: debate is net-destructive on 4/5 benches; StrategyQA is the
  constructive counterexample. Beating round-0 pooled would need per-instance "is this revision
  constructive?", which is the (chance-level) attribution problem of RQ2.
- **Generalization is under way and pre-registered.** Stage 1 (3 model-family ensembles × 5 benches,
  N=3) gives H1 15/15, H2 14/15, H3 7/15 (regime-structured). Stage 2 extends to 4 more benchmarks, a
  strong 14B anchor (`E_strong`), and N=5 (`E_n5`); every cell's disposition is locked in advance
  (`E25_newbench_prereg.md`). The persona⊥backbone tie and N-scaling are the axes being closed; the
  claims to replicate are RQ1 extinction, round-0-vs-final degradation, the provenance→correctness
  signal, and the provenance-selector gain.
- **All selection numbers are offline replays** of a frozen destructive-protocol debate; an end-to-end
  live non-destructive protocol is future work.
- **Hazard detector has no high-precision operating point** at 12% prevalence (RQ2) — it ranks, it does
  not cleanly threshold.

---

> **GPU protocol.** Debate generation runs through a **dedicated GPU-pinned** Ollama instance, one per
> card, to avoid disturbing the other user's job:
> `CUDA_VISIBLE_DEVICES=1 OLLAMA_VULKAN=0 OLLAMA_HOST=127.0.0.1:11435
> OLLAMA_MODELS=/usr/share/ollama/.ollama/models ollama serve` pins **gpu1**; the mirror with
> `CUDA_VISIBLE_DEVICES=0 … OLLAMA_HOST=127.0.0.1:11436` pins **gpu0**, and both can run in parallel
> (RQ6 Stage 2 uses both). Pinning the client is **not** enough — the system server on :11434 is
> unpinned, and `OLLAMA_VULKAN=0` is required or Ollama re-enumerates the other card via Vulkan. Verify a
> test load lands on the intended `pci.bus_id` (gpu0 = `01:00.0`, gpu1 = `03:00.0`), and always launch
> detached (`setsid nohup … < /dev/null &`) so a closing session can't kill it. Whichever card the other
> user is on, **yield it — never preempt.** Zero-GPU analyses run under
> `CUDA_VISIBLE_DEVICES="" /home/alex/venvs/env/bin/python`.
