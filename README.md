# Not All Consensus Is Equal
### Epistemic Provenance of Agreement in Multi-Agent LLM Debate

> Multi-agent debate (MAD) is widely used to make LLM answers more reliable. We show it routinely
> **destroys** correct answers it already found — and that the *origin* of agreement carries correctness
> signal beyond its magnitude and beyond agent-level influence. Preserve every hypothesis, read its
> **provenance**, and re-select: reliability improves at **zero extra LLM calls**.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Backend" src="https://img.shields.io/badge/LLM%20backend-Ollama-black">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20code-orange">
</p>

This repository contains the full code, data, debate logs, results, and manuscript for the journal
extension of our conference work on influence in multi-agent debate (*IA-MAD*). The core results are
entirely reproducible from local, open-weight models served through [Ollama](https://ollama.com) — no paid
APIs. One optional robustness axis (a frontier-scale ensemble, `E_frontier`) additionally uses an
OpenAI-compatible cloud API (DeepSeek); it is not required for any headline result.

---

## Table of contents
- [The idea in one minute](#the-idea-in-one-minute)
- [Key findings](#key-findings)
- [Results at a glance](#results-at-a-glance)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [Hardware & GPU serving](#hardware--gpu-serving)
- [Configuration](#configuration)
- [The manuscript](#the-manuscript)
- [Citation](#citation)
- [License](#license)

---

## The idea in one minute

A debate is a small society of LLM agents that answer a question independently, then read each other and
revise over a few rounds. The deployed system takes a **majority vote over the final round**. We ask a
sharper question than "how many agents agree?": **where did that agreement come from?**

Three observations drive the paper:

1. **Debate forgets.** A correct answer is often generated early and then *socially erased* before the
   final vote — *correct-hypothesis extinction*.
2. **Targeting the failure doesn't work.** We can *predict* which debates will fail (a hazard detector),
   but *which debate benefits from intervention* is unidentifiable from the observable state — across
   **five pre-registered attempts** — despite a large oracle ceiling.
3. **So we fix it structurally.** Instead of overwriting hypotheses, **preserve** all of them and attach a
   **provenance** record (how many agents produced it *independently*, how long it persisted, when it first
   appeared). The central result: at **matched total support**, *independently generated* agreement carries
   far more correctness signal than *socially adopted* agreement.

The whole pipeline — **Preserve → Track Provenance → Select** — runs *post hoc* on stored transcripts and
adds **no** LLM inference.

---

## Key findings

| # | Contribution | Headline number |
|---|---|---|
| **C1** | **Correct-hypothesis extinction is real.** A correct answer present during debate is gone by the final vote. | Recall $0.636 \to 0.760$; **+12.5pp** forgotten; debate is *net-destructive on 4/5 benchmarks*. |
| **C2** | **Failure prediction $\neq$ benefit prediction.** Extinction is detectable and has a large oracle ceiling, but intervention benefit is not identifiable from the pre-intervention state. | Hazard ROC **0.757**; oracle **+16.5pp**; **5 pre-registered negatives**. |
| **C3** | **Hypothesis preservation exposes a large, cheap headroom.** | Ceiling **+25pp** over deployed majority; pool $\approx 3$ candidates/question. |
| **C4** | **Consensus is not epistemically homogeneous (the headline).** At matched total support, independent origin beats *socially adopted* support — even within the same question. | $\Delta\beta_{\text{ind}-\text{adopt}} = 0.96\,[0.88,1.05]$ (genuine adoption $\approx 0$); within-question $\beta_{\text{ind}} = 0.47$; matched-support $\beta_{\text{ind}} = 0.589$; **7/8** benchmarks, **5/5** ensembles (3B–14B, $N\in\{3,5\}$). |
| **C5** | **A zero-cost provenance selector** beats the deployed majority *and* the conference influence vote — but conditionally. | **0.528 → 0.564** (+3.5pp), influence vote *loses* (0.502); significant in 14/40 cells, no significant loss anywhere. |

---

## Results at a glance

**Extinction / preservation ceiling** (pooled, 5 benchmarks × 5 seeds; cluster-bootstrap 95% CI over
questions):

| Benchmark | Majority acc | Correct in final round | Correct anywhere in history | Forgotten (ΔCR) |
|---|---|---|---|---|
| GSM8K | 0.760 | 0.824 | 0.920 | +0.096 |
| BBH | 0.466 | 0.602 | 0.798 | **+0.196** |
| StrategyQA | 0.604 | 0.750 | 0.908 | +0.158 |
| CommonsenseQA | 0.664 | 0.806 | 0.936 | +0.130 |
| TruthfulQA | 0.148 | 0.196 | 0.240 | +0.044 |
| **Pooled** | **0.528** | **0.636** | **0.760** | **+0.125** |

**Not all consensus is equal** — correctness at *matched total support*, varying only the number of
*independent* (round-0) origins:

| Total support | 0 indep | 1 indep | 2 indep | 3 indep |
|---|---|---|---|---|
| 4 | 0.21 | 0.42 | 0.63 | 0.66 |
| 5 | 0.27 | 0.47 | 0.71 | 0.76 |
| 6 | — | 0.53 | 0.77 | 0.87 |

**Inference-time selectors** (gold-free, over frozen logs, zero extra LLM calls):

| Selector | Pooled acc | Δ vs. majority |
|---|---|---|
| Final majority (deployed) | 0.528 | — |
| Conference influence vote | 0.502 | −0.026 |
| Round-0 (independent) majority | 0.560 | +0.032 |
| **Provenance selector (ours)** | **0.564** | **+0.035** |
| Oracle (history-pool ceiling) | 0.760 | +0.232 |

---

## Repository layout

```
influence-mad/
├── src/                       # Library code
│   ├── agents/                #   PeerAgent, personas, INIT/REFINE prompts, Ollama client
│   ├── data_collection/       #   run_debate: synchronous N-agent debate + logging
│   ├── influence_estimation/  #   adoption matrix + PageRank influence score
│   ├── influence_analysis/    #   good/bad/low-influence classification, correlations
│   ├── influence_voting/      #   majority / influence-based / influence-aware voting
│   └── utils/                 #   robust answer extraction, scoring, seeding, I/O
├── experiments/               # One script per experiment (E1–E32) + runners
│   ├── run_multiseed.py       #   generate debates for a benchmark across seeds
│   ├── run_pipeline.py        #   single-seed collect → analyze
│   ├── run_analysis.py        #   influence analysis + plots for one benchmark
│   └── e*.py                  #   the numbered experiments (see "Reproducing the paper")
├── config/                    # YAML configs (agents, models, rounds, seeds, benchmarks)
├── scripts/                   # Dataset download + Ollama setup helpers
├── data/
│   ├── raw/ processed/        #   benchmark questions
│   └── debate_logs/           #   frozen debate transcripts (JSONL) — the analysis input
├── results/
│   ├── tables/                #   *.json result tables (one per experiment)
│   ├── figures/               #   paper figures (PNG)
│   └── logs/                  #   run logs (large Ollama server dumps are git-ignored)
├── paper/                     # LaTeX manuscript, figure scripts, pre-registrations
├── tests/                     # Unit tests
└── requirements.txt
```

---

## Installation

**1. Python environment** (Python 3.10+). The committed `venv/` is git-ignored — create your own:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Ollama** (the local model backend). Install from [ollama.com](https://ollama.com), then pull the
default 3-agent ensemble:

```bash
ollama pull qwen2.5:3b-instruct    # "careful"
ollama pull phi4-mini              # "skeptic"
ollama pull llama3.2:3b            # "bigpicture"
# (larger ensembles, optional) qwen2.5:7b/14b, mistral:7b, llama3.1:8b, gemma2:9b, nomic-embed-text
```

**3. Benchmarks** — download and cache the datasets:

```bash
python scripts/download_benchmarks.py
```

> **Note:** all analysis experiments read the **frozen debate logs already committed in
> `data/debate_logs/`**, so you can reproduce every table and figure *without a GPU* and without
> regenerating debates. You only need Ollama to collect new debates.

---

## Quick start

```bash
# --- Reproduce the headline results (zero-GPU: reads frozen logs, writes results/tables/*.json) ---
python experiments/e18_hypothesis_preservation.py   # C1/C3  extinction + preservation ceiling
python experiments/e22_provenance.py                # C4     provenance predicts correctness
python experiments/e26_independence.py              # C4 ★   NOT ALL CONSENSUS IS EQUAL (headline)
python experiments/e23_provenance_selector.py       # C5     zero-cost provenance selector

# --- Regenerate all paper figures ---
python paper/make_figures.py

# --- Collect fresh debates (needs Ollama running) ---
python experiments/run_multiseed.py --benchmark gsm8k                 # 5 seeds, full benchmark
python experiments/run_multiseed.py --benchmark gsm8k --backend mock --n_samples 5   # fast smoke test, no GPU
```

Each experiment prints its headline numbers and writes a JSON table to `results/tables/`.

---

## Reproducing the paper

Every experiment is a standalone script. The analysis scripts (E1–E27, except the debate generators) are
**zero-GPU** and read the frozen transcripts in `data/debate_logs/`.

| Script | Question it answers | Paper |
|---|---|---|
| `e1_cascade_headroom.py`, `e1e4a_clean.py` | How much correct accuracy is *forgotten*? | §4 (main) |
| `e4a_extinction_anatomy.py` | Social adoption vs. self-error; surface adoption | App. (anatomy) |
| `e2_mechanism_influence.py` | Influence *cannot* attribute the extinction mechanism (AUC ≈ 0.49) | §5 / App. (influence) |
| `e3_router.py` | Round-0 **hazard detector** for extinction (ROC 0.757) | App. (failure≠benefit) |
| `e5_counterfactual.py` | Mechanism-agnostic re-query **rescues** failing debates (+16pp) | App. |
| `e11_benchmark_expansion.py`, `e11_layer2.py` | Extinction detectability across benchmarks | App. |
| `e13_benefit_aware.py` | **Failure ≠ benefit**: oracle +16.5pp, learned CATE ≈ random | §5 / App. |
| `e14_benefit_semantic.py`, `e15_active_probe.py`+`e15_eval.py`, `e16_probe.py`+`e16_eval.py`, `e17_trajectory_precheck.py` | The **five pre-registered negatives** | App. |
| `e6_controller.py`, `e6_offline.py`, `e7_baselines.py`, `e8_ablation.py`, `e9_stats.py`, `e12_threshold.py` | Hazard-guided controller battery (demoted) | App. (controller) |
| `e18_hypothesis_preservation.py` | **Preservation ceiling**: recall 0.636 → 0.760 | §5.2 (main) |
| `e19_selector_bakeoff.py` | Cheap gold-free selectors beat the deployed majority | App. (selectors) |
| `e20_arith_verifier.py` | A deterministic arithmetic verifier does *not* help | App. |
| `e21_semantic_selector.py` | An external LLM judge is a *liability* under capability mismatch | App. |
| `e22_provenance.py` | **Provenance predicts correctness** (n_indep: 0.07 → 0.83) | §5.5 (main) |
| `e23_provenance_selector.py` | **Provenance selector** beats majority & influence vote, zero cost | §5.7 (main) |
| `e24_convergence_selector.py` | Convergence-gating recovers only the constructive regime | App. |
| `e25_generalization.py`, `e25_newbench_verdict.py`, `e25_scale_verdict.py` (`--ext` → 24B + frontier) | Generalization across models (**3B–24B** open-weight + a frontier API), tasks, and $N\in\{3,5\}$ | §5.8 (main) |
| `e26_independence.py` | ★ **Not all consensus is equal** (β_ind, matched support) | §5.6 (main) |
| `e27_surface_genuine.py` | Origin matters, not the convergence-quality of later adoption | App. |
| `e28_h3_robust.py` | H3 robustness: true-adoption decomposition + within-question control | App. |
| `e30_reviewer_robustness.py` | Collinearity / ridge-path / ablation robustness battery | App. |
| `e31_self_consistency.py` | Compute-matched **self-consistency** control (independent sampling beats debate) | §5 / App. |
| `e32_multiplicity.py` | Benjamini–Hochberg FDR over the 40-cell generalization grid | App. |

To collect debates for an alternate ensemble (e.g. a 14B anchor) or agent count, see the `--gen` / `--verdict`
flags documented at the top of `experiments/e25_generalization.py`.

---

## Hardware & GPU serving

All debates in the paper were generated on a workstation with **2× NVIDIA RTX 5060 Ti (16 GB each)**, served
through Ollama. Because the machine is shared, each run is **pinned to the idle GPU** by launching a dedicated
Ollama server bound to that device and a private port, so it never disturbs a co-tenant job:

```bash
# GPU 1  → port 11435   (use CUDA_VISIBLE_DEVICES=0 and :11436 for GPU 0)
CUDA_VISIBLE_DEVICES=1 OLLAMA_VULKAN=0 GGML_VK_VISIBLE_DEVICES=-1 \
  OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
  ollama serve
```

Then point the client at that port via `debate.base_url` in the config (e.g. `http://localhost:11435`). Each
ensemble is sized so all of its agents co-reside on a single 16 GB card with no swapping (e.g. the 14B-anchor
ensemble ≈ 13.5 GB, the five-agent ensemble ≈ 15.5 GB). Any single CUDA-capable GPU with ≥ 8 GB is enough for
the default 3B ensemble; the analysis scripts need no GPU at all.

---

## Configuration

Debates are driven by a YAML file (default `config/config.yaml`):

```yaml
agents:
  - {name: careful,    model: qwen2.5:3b-instruct, persona: careful,    temperature: 0.3}
  - {name: skeptic,    model: phi4-mini:latest,    persona: skeptic,    temperature: 0.5}
  - {name: bigpicture, model: llama3.2:3b,         persona: bigpicture, temperature: 0.5}
debate:   {max_rounds: 3, backend: ollama, base_url: "http://localhost:11434", max_tokens: 512}
influence:{algorithm: pagerank, damping_factor: 0.85}
seeds:    [42, 43, 44, 45, 46]
```

- **Agents** — persona + backbone model + sampling temperature. The five paper ensembles (3B → 14B,
  $N\in\{3,5\}$) are defined in `experiments/e25_generalization.py`.
- **`base_url`** — point this at your pinned Ollama server (see above).
- **Prompts** — persona, debate, intervention, and probe prompts live in `src/agents/peer_agent.py`,
  `experiments/e5_counterfactual.py`, and `experiments/e15_active_probe.py` (and are reproduced verbatim in
  the paper appendix).

**Optional — frontier / cloud API (`E_frontier`).** The one robustness axis that is not local uses an
OpenAI-compatible endpoint (DeepSeek). It is *not* needed for any headline number; the frozen logs are
already committed. To regenerate it, set `backend: openai` and provide a key **from outside the repo** (never
commit it):

```bash
export DEEPSEEK_API_KEY="$(cat ~/.deepseek_key)"   # chmod 600, outside the repo
export DEEPSEEK_MODEL=deepseek-v4-flash
python scripts/e25_frontier_gen.py --smoke          # small sanity check first
```

---

## The manuscript

The LaTeX source is in [`paper/`](paper/):

- `paper/paper.tex` — the self-contained manuscript (Introduction → Conclusion + appendices).
- `paper/references.bib` — bibliography.
- `paper/make_figures.py` — regenerates every figure in `results/figures/` from the frozen tables.
- `paper/*_prereg.md` — the pre-registrations for the benefit-prediction attempts.

Build with any LaTeX toolchain, e.g. `latexmk -pdf paper/paper.tex` (ensure `graphicx` and the appendix
counters are available; a minimal preamble wrapping `paper.tex` is all that's required).

---

## Citation

If you use this code or the findings, please cite:

```bibtex
@article{notallconsensus,
  title  = {Not All Consensus Is Equal: Epistemic Provenance of Agreement in Multi-Agent LLM Debate},
  author = {<authors>},
  year   = {2026},
  note   = {Journal extension of the conference work on influence in multi-agent debate}
}
```

---

## License

Released under the [MIT License](LICENSE). Replace the copyright-holder line in `LICENSE` with the
author name(s) before publishing.

## Acknowledgements

Built on open-weight models served locally through [Ollama](https://ollama.com) (Qwen2.5, Phi-4-mini,
Llama-3.2/3.1, Mistral, Gemma-2) and standard scientific-Python tooling (NumPy, pandas, scikit-learn,
Matplotlib). Benchmarks: GSM8K, StrategyQA, CommonsenseQA, BBH, TruthfulQA, AQuA, MMLU, and ARC.
