# Pre-registration — E3b-v3 frozen external replication on MATH

*Frozen 2026-08-26, BEFORE any MATH debate generation, scoring, or E3b effect computation.*

## Hypothesis
**H-E3b:** On open-ended reasoning tasks, pre-revision debate dynamics carry
social-provenance information about subsequent attribution **beyond temporal
persistence**. Operationally: predicting the M0-residual of the continuous
attribution score `SA` from dynamics-only features gives Spearman ρ > 0, exceeding a
within-domain permutation null. GSM8K (the discovery domain) shows residual ρ ≈ 0.40.

## Frozen MATH selection rule (blind to any E3b effect)
Source `EleutherAI/hendrycks_math`, **all 7 categories** (algebra, counting_and_probability,
geometry, intermediate_algebra, number_theory, prealgebra, precalculus), **test** split.
Include a problem iff **all** hold:
1. `level ∈ {Level 3, Level 4}` (medium-hard: excludes trivial L1–2 and olympiad L5);
2. the gold `\boxed{}` answer normalizes to a **bare number** (integer/decimal via
   `_extract_number`) — a *scoring-validity* filter so `is_correct` is reliable, NOT an
   effect filter;
3. deterministic order: **fixed-seed (42) shuffle** of the filtered pool (category-mixed,
   reproducible), take the first N.
No category is selected on; the difficulty band is fixed a priori; **no re-selection based
on results** (no algebra→geometry→number-theory hunting).

## Effect-blind pilot gate (pilot may inspect ONLY these four)
Pilot N = 40 questions, seed 42. Allowed metrics: majority accuracy, extinction-event
yield, scoring/parser validity, runtime. **The pilot MUST NOT compute any E3b ρ.**
- **GO (full replication)** iff `0.30 ≤ Acc_majority ≤ 0.80` AND `extinction_yield ≥ 5%`
  (ideal 45–70% acc, 8–15% yield).
- **STOP** if SVAMP-like (`Acc > 0.90`, yield ≈ 0) → do not spend GPU.

## Frozen replication protocol (only if pilot GO)
Run `e3b_v3_attribution.py` with MATH added as a held-out domain. **Nothing is tuned on
MATH**: identical encoder (nomic-embed-text), residualization, M0/M1/M2 feature sets,
Ridge/hyperparameters, within-bench z-scoring, target `SA`, scoring.
**Primary test:** `ρ(DYN → SA_residual-of-M0)` on MATH vs its within-domain permutation null.
- **Replicated** if MATH residual ρ ≳ 0.25 and exceeds the permutation null (like GSM8K).
  → Claim: *in open-ended reasoning, revision dynamics carry social-provenance information
  beyond temporal persistence* (no longer a social/self classifier).
- **Not replicated** if MATH residual ρ ≈ 0 within null → close E3b: GSM8K effect does not
  generalize; attribution-beyond-persistence unproven. Proceed to E6 (E3a + E5 independent).

## Secondary (run later, resources permitting)
AQuA as a **negative-control domain** (closed 5-way MC): pre-registered prediction
`ρ_incremental,AQuA ≈ 0` (surface coincidence, like CommonsenseQA). A pattern of
GSM8K≈+0.40, MATH≈+0.3, AQuA≈0, CSQA≈0 would make the result a *regime-specific* mechanism
(open-ended vs closed answer space), stronger than a universal claim.
