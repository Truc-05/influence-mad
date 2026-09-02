# E15 — Pre-registration: Active Diagnostic Probe (GO / KILL gate)

**Status:** LOCKED before running. 2026-08-27, after E13 (failure≠benefit) and E14
(passive semantic rescue FAILED far from gate).

**Hypothesis.** The treatment benefit is unidentifiable from the *passive* pre-intervention
state (E13/E14). But a cheap **active** probe that *acquires new information* — asking each
agent to self-diagnose its answer before any revision — may reveal intervention
susceptibility. Formally the controller becomes
`X0 → P → τ̂(X0,P) → intervene iff τ̂ > c`, where `P` is probe evidence. This is
categorically different from E14 (which only re-represented the same transcript).

## Probe (LOCKED — no prompt-shopping)

Per agent holding a candidate answer at round 0, ONE cheap single-turn generation
(no answer revision), same persona/model, gpu1-pinned :11435:

> {persona}
> Question: {question}
> Your current answer: {own_answer}
> Your reasoning: {own_reasoning}
> Before any revision, critically examine your OWN answer and name the single most likely
> specific flaw that could make it incorrect. Do NOT change your answer.
> Respond exactly:
> FLAW: <one specific flaw, or "none" if confident there is no error>
> CONFIDENCE: <0-100 how confident your answer is correct>

## Probe features (LOCKED), aggregated order-invariantly over the 3 agents

- self-reported `CONFIDENCE` (0-100): mean / min / max / std
- `found_flaw` (FLAW not none/empty): count and fraction over agents
- flaw-text length: mean / max
- semantic: residualized nomic-embed of the FLAW text — own content, and cosine(flaw,
  own round-0 reasoning) (did the agent engage or deflect); mean/min/max
The controller may use `[X0_cheap, probe_features]`. Same S- and T-learner as E13/E14,
GroupKFold(5) by qid, no model search.

## Baselines (all at matched budget, same population, same ITE labels)

Never · Always · Random · Hazard (E3a) · Semantic-CATE (E14) · **Probe-CATE** · Oracle(true ITE).

## Gate (LOCKED)

- **PRIMARY (decision, binding):** `V_Probe(b) − V_Random(b)` cluster-bootstrap (by qid,
  B=4000) 95% CI **lower bound > 0** in the pre-specified **20–40%** budget band.
- **SECONDARY (mechanism, corroborating):** `AUC_ITE-sign > 0.65` OR `ρ(τ̂,ITE) > 0.20`.
- **PASS = PRIMARY met.** Report SECONDARY as evidence the win is real information, not luck.

## Stop rule

- **PASS →** proceed to E16 (probe ablation: diagnostic vs **token/call-matched PLACEBO
  probe** vs generic-reflection vs no-probe) to prove it is *information acquisition*, not
  just extra LLM compute; then E17 value-of-information; then E11 6-bench, E10 models,
  crossed persona×backbone. The probe controller becomes the paper's algorithmic hero.
- **FAIL →** STOP method development. Freeze the paper on E13/E14: *failure is predictable,
  intervention benefit is not identifiable from the pre-intervention state (passive or
  actively probed), despite a large oracle ceiling.* No further probe/prompt/classifier
  shopping.

## What we will NOT do

No classifier search to rescue E14; no threshold tuning; no multiple embedding models; no
6×6 model×benchmark before E15 locks; no adding samples to force E6 significance.
