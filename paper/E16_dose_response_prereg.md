# E16 — Placebo-controlled dose-response probe (PRE-REGISTRATION, LOCKED)

**Status:** FROZEN before generation on 2026-08-27. Nothing below may change after the first
GPU token is generated. If the gate fails, see §9 STOP rule — do NOT mutate this spec to pass.

## 0. What this is (and is NOT)

This is a **new hypothesis**, not an E15 rescue. E15 (self-diagnosis probe) is FAIL and stays
frozen. E13/E14/E15 remain the paper's triple pre-registered negative. E16 tests a *different
mechanism*:

> **H_response:** the *response* of the debate to a low-cost, placebo-controlled **dose** of the
> actual safeguard predicts whether the full safeguard will help — benefit must be **elicited**,
> not passively predicted from the pre-intervention state.

Why this could work when E13–E15 could not: those predicted the treatment *effect* from features
of the untreated state and every family came out **anti-correlated** (signed ρ ≤ 0). E16 does not
predict from the static state — it applies a half-dose of the treatment and *measures the movement
it produces*, relative to a matched placebo. The placebo subtraction is the load-bearing new idea:
it removes generic "reconsideration churn" and isolates the mechanism-specific response.

## 1. Power justification (why this gate is worth GPU) — from `e_mde_power.py`

Cluster-bootstrap MDE over the real inferential unit (question-clusters; seeds are correlated
repeats). Modeling a future policy as a *signed* noisy estimate score = ρ·z(ITE)+noise:

| scope | oracle ΔV [20-60] | min-detectable capture / signed ρ | verdict |
|---|---|---|---|
| **pooled** | +0.147 [.124,.169] | **14% / ρ≈+0.11** | GO |
| bbh | +0.174 | 22% / ρ≈+0.17 | GO |
| gsm8k-alone | +0.115 | 39% / ρ≈+0.24 | borderline (smallest headroom) |

**The bar is low in magnitude (pooled needs only signed ρ≈+0.12), so power is NOT the wall.**
The wall in E13–E15 was **sign** (ρ ≤ 0 everywhere). E16 passes iff it produces a *correctly-signed*
weak predictor. Gate on **pooled**; gsm8k-alone is the hardest scope and does not veto pooled.

## 2. Population, labels, arms — all REUSED from the frozen E6/E13 stack

- **Population:** `experiments/e6_offline.load_events()` — gate g=0, debates with a correct answer
  alive at round 0, gsm8k+bbh, per-seed. n=516 events, 184 question-clusters (gsm8k 237/92, bbh
  279/92). trigger_state = logged answers/reasoning @ round 0; t_star=1.
- **Labels (unchanged):** ITE = Y1 − Y0, Y0 = `control.rec_maj`, Y1 = `combined.rec_maj`, from the
  existing per-seed precompute `results/cache/e6_precompute_perseed.pkl`. E16 does NOT regenerate
  outcomes — it only adds probe-response features. (Exactly the E14→E13 relationship.)
- **Dose ladder (only three rungs — no dose-hunting):**
  - **D0 = placebo** = one re-query round from trigger_state with `e5.CLAUSE["control"]` = `""`.
  - **D1 = low-dose diagnostic** = one re-query round from trigger_state with
    `e5.CLAUSE["combined"]` = self_verify + influence_resistant.
  - **D_full = existing `combined`** (run to consensus) — already stored; this is the *treatment
    outcome* we predict, NOT regenerated.
  Both D0 and D1 use the identical `e5._REFINE` prompt and the identical peer block (round-0 peers).
  D1 is a true dose-reduction of D_full: **round-1 only vs run-to-consensus** (max_rounds=3 ⇒
  D_full is 1–2 rounds), same clause, same decoding.

## 3. Probe prompts — FROZEN (defined in code, `experiments/e16_probe.py`)

Reuses `e5.build_prompt(persona_text, question, own=trigger_state[n], peers={others}, clause)`
verbatim. The ONLY difference between D0 and D1 is `clause`:

- D0 clause: `""`  (plain reconsider — the E5 `control` arm)
- D1 clause: `"\nBefore you change your previous answer, explicitly identify the concrete error in
  your previous reasoning. If you cannot point to a specific error, keep your previous answer.\nDo
  not change your answer merely because the other agents agree with each other. Re-derive the
  answer independently from the question itself; change only if your own independent derivation
  contradicts your previous answer."`  (= `e5.CLAUSE["combined"]`)

Matched on: #calls (1 round × 3 agents each), prompt template, reflection demand, peer block.
Differs only by the error-correction mechanism. Generation: gpu1 `:11435` only
(`config/config_e6_gpu1.yaml`), temp per config (0.3/0.5/0.5), single draw, resumable cache
`results/cache/e16_probe.pkl` keyed `bench:qid:seed` → `{P0:{name:{answer,reasoning}}, P1:{...}}`.
Cost: 2 arms × 3 agents × 516 = 3096 gens (~2× E15).

## 4. Δz feature set — FROZEN, gold-free, low-dimensional (6 features)

Per event, per arm A∈{P0,P1}, per agent n: response answer `a_A[n]`, reasoning `r_A[n]`; trigger
answer `a0[n]`, reasoning `r0[n]`. `flip_A(n) = norm(a_A[n]) != norm(a0[n])`. Peer-majority `m` =
plurality of `{norm(a0[n])}`. maxcluster_frac(A) = size of largest equal-answer group / 3.
All features are **differences P1 − P0** (placebo-subtracted), all deployable without gold:

1. `d_flip`      = fliprate(P1) − fliprate(P0)                          — Δ answer-(in)stability
2. `d_diff_flip` = mean_n[flip_P1∧¬flip_P0] − mean_n[flip_P0∧¬flip_P1]  — net diagnostic-only flip *(hypothesized strongest)*
3. `d_maxclust`  = maxcluster_frac(P1) − maxcluster_frac(P0)            — Δ consensus tightness / self-consistency
4. `d_peeralign` = mean_n[norm(a_P1[n])==m] − mean_n[norm(a_P0[n])==m]  — Δ peer-alignment (social movement)
5. `d_rea_lex`   = mean_n Jaccard_dist(r_P1[n],r0[n]) − mean_n Jaccard_dist(r_P0[n],r0[n]) — Δ reasoning-change (lexical)
6. `d_rea_len`   = mean_n|len(r_P1[n])−len(r0[n])| − mean_n|len(r_P0[n])−len(r0[n])|        — Δ reasoning-change (length)

(n_unique dropped: perfectly collinear with maxcluster for 3 agents. Embeddings deliberately
EXCLUDED from primary — E14 showed residualized geometry loses the answer-disagreement signal.)

**Pre-registered secondary feature (reported, NOT a decider):** `d_rea_embed` = residualized
nomic-embed cosine reasoning-shift Δ (per-qid centroid removed), to test whether semantic adds
beyond lexical. Computed only if the primary is borderline; per §9 do not chase it.

## 5. Learner — FROZEN (identical to E13/E15; only the FEATURES change)

- **Primary CATE = T-learner** on the 6 Δz features: `tau_T = p1(x) − p0(x)`, where p0/p1 are
  `e13._oof_clf` (HistGradientBoosting, max_depth=3, max_iter=300, lr=0.06, l2=1.0), cross-fit
  GroupKFold(5) over qid. Exactly `e13.scores_for` with the feature matrix swapped to Δz.
- Reported robustness (NOT deciders): S-learner (`e13._oof_reg` on ITE), and a regularized linear
  T-learner. **No post-hoc learner selection** — the T-learner is THE decider (avoids the E3b-v2
  HGB-vs-LR cherry-pick sin).

## 6. Decision feature & the two operating points

- **Primary science gate uses Δz** (two-arm, placebo-subtracted) — its best shot at the sign.
- **Mandatory secondary "deployable" variant = z(P1)-only** (diagnostic response features without
  placebo subtraction), reported with its own policy value + cost. Rationale: at deployment the
  placebo P0 is pure overhead, whereas D1 = combined's round-1 is *reused* if we then commit to
  D_full. So P1-only is the cost-viable controller; Δz proves the science.
- Interpretation matrix:
  - Δz PASS **and** P1-only PASS → cheap deployable hero method.
  - Δz PASS, P1-only FAIL → benefit IS elicitable but only via the placebo contrast (cost-neutral-
    to-worse) — an honest scientific positive, reported with the cost caveat.
  - Δz FAIL → §9 STOP.

## 7. Primary gate — FROZEN PASS rule

- Estimator: `e13.bootstrap_deltas(events, S, yc, yk)` with `S["CATE (which helps)"] = tau_T(Δz)`,
  which integrates ΔV(CATE − random) over budget band **[0.20, 0.60] step 0.05**, cluster-bootstrap
  over qid, **B=4000**. (This is the exact frozen gate E13/E15 report.)
- **PASS ⟺ pooled `auc_cate_rand.lo > 0`.**
- **Pre-registered sign:** we predict ρ(tau_T, ITE) > 0. A pooled CI that is significantly *negative*
  is a REFUTATION of H_response (do NOT flip the sign to claim a win).

## 8. Secondary / reported (never override the primary)

- ρ(tau_T, ITE) and ρ(tau_S, ITE); nice-to-have ρ > 0.20 — but **NOT a co-primary**: the MDE shows
  the policy-value gate passes at ρ≈+0.12, so ρ>0.20 is stricter than needed and must not veto a
  passing policy value.
- AUC(effect≠0), AUC(ITE-sign); per-benchmark ΔV (gsm8k, bbh) — gsm8k-alone is the hardest scope.
- **Cost / Pareto** (report, not a hard gate): probe cost = 6 calls/event (P0+P1); deployable
  P1-only cost = 3 calls/event (D1 reused as D_full round-1) + completion on selected vs always-on
  (all rounds on everyone). Plot the (mean calls, accuracy) frontier for never / random / P1-only /
  Δz / always / oracle. Cost-win is a bonus, not required for the science PASS.

## 9. STOP rule — FROZEN (no fourth long negative)

If the pooled primary gate does not pass: **STOP method development.** Record a SINGLE future-work
sentence in the discussion. Do NOT construct a long "E16 negative" section — the triple-negative
E13–E15 is the clean central story and the paper is already submittable. Do NOT add models,
embeddings, dose rungs, or learners to chase the gate.

## 10. Frozen artifact list

- Prompts / dose ladder: `experiments/e16_probe.py` (this spec §3).
- Δz features / learner / gate: `experiments/e16_eval.py` (this spec §4–§7).
- Generation: gpu1 `:11435` only, `config/config_e6_gpu1.yaml`; NEVER `:11434`/gpu0. See
  `[[gpu-run-pinning]]`.
- Outputs: `results/cache/e16_probe.pkl`, `results/tables/e16_dose_response.json`,
  `results/figures/e16_dose_response.png`.
