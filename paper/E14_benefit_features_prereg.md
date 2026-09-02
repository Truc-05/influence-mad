# E14 — Pre-registration: can RICHER INFORMATION make intervention benefit predictable?

**Status:** LOCKED before running. Written 2026-08-27, after E13.
**Runs only AFTER** the E11-Layer-2 StrategyQA verdict lands (do not interrupt StrategyQA).
**One shot.** If the pre-registered gate is not met, we STOP trying to build a benefit
controller and E13 becomes the paper's central contribution. No classifier/model-shopping.

## Motivation (fixed by E13)

E13 established, on the corrected per-seed cache:
- failure `Y0` predictable (OOF AUC 0.73), post-intervention `Y1` predictable (0.67),
- but the treatment effect `τ(x)=E[Y1−Y0|X]` is **not** predictable from round-0 features
  (effect-presence AUC 0.58; sign anti-correlated, ρ(τ̂,ITE) ≤ 0, both S- and T-learner),
- while the **oracle ceiling is large** (+16.5pp pooled / +20.4 BBH over never).

Conclusion driving E14: benefit-targeting is worth it, but the **round-0 state does not
carry the information**. The only legitimate rescue is to CHANGE THE INFORMATION available
to the controller — not the classifier.

## Hypothesis

There exists a cheap, deployable information source `Z` (beyond round-0 handcrafted
features) such that `E[ITE | X, Z]` is predictable well enough to beat a random budget.

## Information sources (each is a DISTINCT pre-registered arm; run in feasibility order)

| # | Source | New GPU? | Reuses existing ITE labels? | Timing caveat |
|---|---|---|---|---|
| **B2** semantic | frozen embeddings of round-0 reasoning (own / peer / deltas), via `nomic-embed-text` | embed only (cheap) | **YES — drop-in** | none (same g=0 gate) |
| **B3** probe | cheap pre-intervention diagnostic ("before revising, name one concrete error"); its output → features | YES (1 gen/agent) | needs a small new precompute | none (probe precedes the full intervention) |
| **B1** mid-debate | round-1 dynamics (answer changes, disagreement trajectory, support shift, revision velocity, peer-alignment change) | **YES — new g=1 precompute** | **NO** — outcomes must be recomputed for a g=1 gate | intervention would start at round 2 → misses t*=1 deaths; also round-1 factual state is partly what the intervention changes (leakage risk) |

**Feasibility-driven order (differs from the intuitive 1→2→3):**
1. **B2 semantic FIRST** — it is a pure drop-in: same g=0 gate, same observed ITE labels,
   only the feature vector `X` is augmented with round-0 reasoning embeddings. This cleanly
   isolates "does richer round-0 *content* (vs handcrafted geometry) predict benefit?" with
   ZERO new counterfactuals. Note E3b-v2 found nomic-embed cosine on same-question reasoning
   is topic-dominated → **must residualize** (subtract per-event centroid) before use.
2. **B3 probe** — the theoretically strongest (adds genuinely new signal about the
   intervention itself), but needs GPU (one probe generation per agent) and a small precompute
   linking probe output → ITE. Run only if B2 shows partial signal or as the designated
   strongest single shot.
3. **B1 mid-debate** — requires a separate g=1 counterfactual precompute and carries a real
   timing/leakage caveat; treat as a distinct sub-study, not a drop-in, and only if B2/B3
   motivate it.

Recommendation for the single pre-registered shot: **B2 (drop-in, cheap, clean) as the
primary; escalate to B3 (probe) as the strongest GPU shot if B2 is borderline.**

## Pre-registered success gate (LOCKED — evaluate on OOF, cluster-cross-fit by qid)

Predictive gate (necessary):
- `AUC_{ITE-sign} > 0.65`  **OR**  `ρ(τ̂, ITE) > 0.20`   (Spearman, OOF).

Decision gate (necessary, the one that matters):
- `V_benefit-aware(b) > V_random(b)` with **cluster-bootstrap (by qid, B=4000) 95% CI
  excluding 0** in the pre-specified **20–40%** budget band (`AUC_{20-40}` of
  benefit−random CI lower bound > 0).

**PASS = both gates met.** Anything else = FAIL.

## Evaluation protocol (reuse E13 machinery unchanged)

- Same per-seed cache, same population, same ITE = Y1 − Y0.
- Learn `τ̂` with the SAME learners as E13 (S-learner + T-learner) on `[X, Z]`; GroupKFold(5)
  by qid. No hyperparameter search beyond E13's settings (no model-shopping).
- Compare policies random / hazard / disagreement / benefit-aware(`τ̂`) / oracle at matched
  budget; report the four headline metrics + policy-value curves, pooled and per-benchmark.
- Report the diagnostic triple AUC(Y0), AUC(Y1), AUC(effect), and ρ(τ̂,ITE), as in E13.

## Outcomes

- **FAIL** → STOP. E13 is the contribution: *"failure risk is predictable, but intervention
  value is not identifiable from the pre-intervention state, despite a large oracle ceiling."*
- **PASS** → novelty upgrade: `detect failure → estimate benefit → select intervention`; the
  paper becomes **benefit-aware adaptive intervention**, not hazard-guided safeguarding.
