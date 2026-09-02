# E25 Generalization — Why H3 holds in 7/15 cells (deep-dive)

**Source:** `results/tables/e25_generalization.json`, run `2026-08-30 15:22` (rc=0, all 45 seed×bench `rc=0`).
**Scope of E25:** 15 cells = 3 debate conditions × 5 benchmarks × 3 seeds. The three conditions vary
**debater capability and model family** at fixed **N=3** (N>3 remains future work):
- `E_orig` = qwen2.5:3b / phi4-mini / … (3B×3, frozen logs reused)
- `E_7b` = qwen2.5:7b / mistral:7b / llama3.1:8b
- `E_mixed` = gemma2:9b / phi4-mini / llama3.2:3b (mixed capability, cross-family)
**Status:** Read-only analysis. Selector **not** modified or re-tuned. This note is the input for the paper's applicability-conditions / generalization section.

H3 test per cell: *does the provenance selector beat final-majority?* → **PASS iff best-selector ΔAcc CI-low > 0** (bootstrap CI, per-cell n below).

---

## 1. Headline: the failures are not scattered — they concentrate by benchmark

| bench | H3 pass | combo_auc (prov signal) | final-maj | CR_history (pool ceiling) |
|---|---|---|---|---|
| commonsenseqa | **3/3** ✅ | 0.86 | 0.67 | 0.94 |
| gsm8k | **2/3** | 0.91 | 0.78 | 0.93 |
| bbh | **2/3** | 0.76 | 0.52 | 0.82 |
| strategyqa | **0/3** ❌ | 0.63 | 0.63 | 0.88 |
| truthfulqa | **0/3** ❌ | 0.69 | 0.17 | 0.28 |

**Partition:**
- Multi-answer reasoning benches (gsm8k / commonsenseqa / bbh): **7 / 9 pass**.
- Degenerate benches (strategyqa / truthfulqa): **0 / 6 pass**.

All 8 H3 failures = the 6 strategyqa+truthfulqa cells + exactly 2 marginal misses among the 9 reasoning cells. The failures have **structure**; this is a *regime* boundary, not a method that works intermittently.

## 2. What predicts H3 across the 15 cells

Point-biserial correlation with H3-pass, and correlation with the selector's ΔAcc CI-low:

| predictor | r with H3-pass | note |
|---|---|---|
| **combo_auc** (provenance→correctness signal) | **+0.70** | corr(CI-low, combo_auc) = **+0.73** — dominant |
| CR_history (pool correctness ceiling) | +0.51 | |
| final-majority | +0.47 | |
| recoverable mass = CR_history − final-maj | +0.30 | weak on its own |
| candidate base-rate | +0.18 | |

Signal strength (combo_auc), **not** headroom, is the primary gate. A simple two-gate rule
`combo_auc ≥ 0.74 AND recoverable-positives ≥ 15` reproduces 13/15 cells; the 2 exceptions are exactly the ceiling and floor cases below.

## 3. Two applicability conditions — and each degenerate bench violates a different one

$$\text{provenance reranking helps} \;\Longleftarrow\; \underbrace{\text{provenance is informative}}_{\textbf{identifiability}} \;\land\; \underbrace{\text{a correct hypothesis exists in the pool}}_{\textbf{availability}}$$

Both are properties of the **task**, not of the selector. StrategyQA mainly violates *identifiability*; TruthfulQA mainly violates *availability*.

### A. StrategyQA → **identifiability failure** (weak provenance discrimination; 0/3, all conditions)
StrategyQA is yes/no. Candidate base-rate of correctness ≈ 0.55–0.58 (near coin-flip), and provenance features barely discriminate correct from incorrect:

| cond | base-rate | n_indep AUC | n_persist AUC | combo_auc | ΔAcc | CI-low |
|---|---|---|---|---|---|---|
| E_orig (n=500) | 0.56 | 0.61 | 0.55 | 0.61 | **−0.012** | −0.038 |
| E_7b (n=100) | 0.58 | 0.58 | 0.60 | 0.62 | −0.020 | −0.060 |
| E_mixed (n=100) | 0.57 | 0.65 | 0.57 | 0.66 | +0.050 | −0.040 |

Provenance carries ~chance information about which of two answers is correct, so this setting offers **weak provenance discrimination** — little leverage for any provenance-based reranking. The E_orig cell is at full power (n=500) and still shows a *negative* point estimate, so this is structural, not sampling noise. (We do **not** claim an impossibility result — only that leverage is limited here.)

### B. TruthfulQA → **availability failure** (low correct-hypothesis availability; 0/3, all conditions)
The oracle over the *entire* debate history tops out at only 24–31% correct — the correct hypothesis is usually absent from the pool, so there is little for any selector to recover:

| cond | n | final-maj | CR_history | combo_auc | ΔAcc | CI-low |
|---|---|---|---|---|---|---|
| E_orig | 500 | 0.15 | 0.24 | **0.76** | +0.020 | 0.000 |
| E_7b | 100 | 0.19 | 0.29 | 0.65 | 0.000 | 0.000 |
| E_mixed | 100 | 0.17 | 0.31 | 0.67 | −0.010 | −0.070 |

Decisive case: **E_orig/truthfulqa has adequate signal (combo_auc 0.76) and full power (n=500) yet still cannot win** — the binding constraint here is availability (answer-mass in the pool), not signal or sample size.

## 4. The two "reasoning-bench" misses are explainable and benign

- **E_7b/gsm8k — ceiling.** 7B debaters already reach final-majority 0.82 (headroom 0.12). Signal is strong (combo_auc 0.92) and ΔAcc is genuinely positive (+0.015) but too small to certify at n=400 (CI-low −0.005). GSM8K passes in E_orig (+0.054) and E_mixed (+0.093); the ceiling only bites when the debaters are strong enough to leave little forgotten-correct mass.
- **E_orig/bbh — marginal CI-graze.** ΔAcc +0.032 (material, positive), CI-low −0.002. Both sibling conditions (E_7b/bbh, E_mixed/bbh) pass. A boundary coin-flip, not a substantive failure.

Both misses have **positive point estimates**.

## 5. No significant accuracy loss where the conditions are unmet (precise claim)

Across all 8 failure cells, **every bootstrap CI includes zero** — no cell shows a statistically significant accuracy *loss* (no CI-high < 0). Point estimates span −0.02 to +0.05; three cells are slightly negative (E_orig/strategyqa −0.012, E_7b/strategyqa −0.020, E_mixed/truthfulqa −0.010) but none significantly so.

> Honest wording: where its applicability conditions are unmet, the selector **neither reliably helps nor significantly hurts**. Do **not** claim it "degrades gracefully to majority" — individual cells do carry small negative point estimates; the defensible statement is only that no loss is significant.

## 6. Paper framing (recommended, safe wording)

Do **not** claim "the selector always wins," and do **not** assert "no reranking method can improve." Claim the conditional, verifiable result:

> **The provenance selector converts a diagnostic signal into accuracy gains under two applicability conditions — identifiability (provenance is informative about correctness) and availability (a correct hypothesis exists in the debate pool). On the three multi-answer reasoning benchmarks where both hold, it reliably beats final-majority (7/9 cells; the two exceptions are a ceiling effect and a CI-boundary graze, both with positive point estimates). These conditions provide limited leverage on the two remaining benchmarks: StrategyQA offers weak provenance discrimination (identifiability), whereas TruthfulQA is constrained by low correct-hypothesis availability. In no cell does the selector significantly reduce accuracy.**

This reframes H3 from a weakness ("passes only 7/15") into a **characterization of the method's applicability conditions** — consistent with H1 (15/15, extinction) and H2 (14/15, provenance signal), and generalizing across model capability and family at N=3.
