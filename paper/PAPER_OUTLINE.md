# Paper write-up spine — RE-LOCKED on epistemic-provenance headline (2026-09-01; 6-beat arc 2026-08-30; C-spine 2026-08-28)

**Title (DECIDED 2026-09-01 — new identity):** *Not All Consensus Is Equal: Epistemic Provenance of
Agreement in Multi-Agent LLM Debate* (sub-theme: *debate forgets correct answers; independent origin,
not influence, tracks correctness*).
*Prior working title (superseded, keep for continuity with old drafts):* *Predicting Failure Is Not
Predicting Benefit: Preservation & Provenance in Multi-Agent LLM Debate.*

**Narrative arc (RE-LOCKED 2026-09-01):**
extinction → influence insufficient → predictability ≠ actionability → preservation → provenance predicts
correctness → **independent vs inherited (why provenance matters)** → downstream decision utility
(reranking, regime-bound) → generalization.
Two structural moves vs the 2026-08-30 lock: (1) the **headline hypothesis changes** — the old H3
("provenance selector beats final-majority", significant in only 16/45 cells) is DEMOTED from a
central contribution to a *"From Mechanism to Decision Utility"* subsection (reported in full, not
hidden); (2) the new **H3 = an epistemic-provenance principle** (E26): *at matched total support,
independently generated agreement carries more correctness signal than socially inherited agreement.*
This is near-universal (7/8 bench, 5/5 config incl 14B & N=5) — a far stronger journal-extension
headline than a regime-bound selector. The conference influence layer stays repurposed as §5
*"influence is insufficient"*; E25 generalization + the selector's applicability conditions
(**identifiability ∧ availability**) live together in the downstream-utility section.

**Target:** ESWA / KBS (Q1). Journal extension of the conference IA-MAD paper.

**One-line thesis.** In multi-agent LLM debate, *when* a correct answer will die is predictable, but
*who benefits* from intervention is not — from any pre-intervention signal we tried. So we stop
targeting intervention and instead stop the loss structurally: preserve every hypothesis and re-select
from provenance. This recovers part of a large, quantified headroom with **no extra LLM calls**.

---

## Hypothesis structure (RE-LOCKED 2026-09-01 — this is the headline framing)

**Three headline hypotheses:**

- **H1 — Correct-hypothesis extinction.** A correct answer that existed during the debate is gone by
  the final vote (E1/E4a/E18).
- **H2 — Provenance predicts hypothesis correctness.** A hypothesis's own debate history (independent
  support, persistence, first round, …) predicts whether it is correct (E22; marginal AUC).
- **H3 — Independent support carries greater epistemic value than socially inherited support, *at
  matched total support*** (E26). Controlling for total consensus, the partial effect of round-0
  independent support is strongly positive: **β_ind = +0.649 [+0.565, +0.737]** (per-vote 2.68×
  inherited: +1.035 vs +0.386); FE-adjusted +0.656; **7/8 benchmarks significant** (only StrategyQA
  n.s.), **5/5 ensembles** incl. E_strong (14B) and E_n5 (N=5). Joint model {ind, inherited, persist}
  keeps all three CIs clear of 0 (ordering ind > persist > inherited).

**Headline principle (the journal-extension novelty — NOT "a new selector"):**
> *Not all consensus is equally informative: independently generated agreement carries more epistemic
> signal for correctness than socially inherited agreement.*

**Old H3 is DEMOTED, not deleted.** "Provenance selector > final-majority" is no longer a
hypothesis/contribution; it moves to a **"From Mechanism to Decision Utility"** subsection. Report it
in full and honestly: the selector beats deployed majority + conference influence-voting at zero cost,
but the accuracy win is **regime-bound** (significant in 16/45 cells; no significant loss anywhere).
This keeps the mixed/negative result visible without letting 16/45 become a central claim that weakens
the paper. The near-universal claim (H2/H3) is the mechanism; the selector is one application of it.

**Logical progression (keep explicit so H3 ≠ a restatement of H2):** H2 establishes provenance *has*
signal; H3 isolates *which component* carries it, by controlling total support. H3 is the **mechanism
behind** H2, not a duplicate.

**Wording guardrails (honor verbatim — 4 locked corrections):**
1. **Keep "at matched total support" in every H3 statement** — it is the load-bearing clause; without
   it H3 collapses into H2. n_independent and n_support are 0.83 collinear, so the control is the claim.
2. **Place the incremental-AUC caveat adjacent to H3.** Adding n_independent over n_support raises AUC
   only **+0.007** (collinearity). Frame H3 as *epistemic-value-per-vote*, NOT *incremental ranking
   power*; this +0.007 is precisely the quantitative reason the downstream selector is regime-bound, so
   the caveat becomes the mechanism→utility bridge, not a weakness a reviewer can weaponize.
3. **Claim "carries more correctness *signal*" not "is more *valuable*/causal."** The result is
   observational (logistic on candidate features); the matched-stratum contrast is causal-flavored but
   still observational, and n_independent partly encodes question difficulty (E22). Report the
   difficulty association as a limitation — and note it is *part of* the point (independent agreement
   marks better-determined questions = wisdom-of-independent-crowds).
4. **Persistence is a supporting result, not in the headline sentence.** Keep H3's headline a single
   contrast (independent vs inherited); mention persistence (joint β_persist +0.448, CI clear of 0) as
   "a second, weaker provenance dimension." Also: report StrategyQA as the honest boundary (7/8), and
   for the demoted selector never say "degrades gracefully / doesn't hurt" — say "no fail cell shows a
   *significant* loss (every CI includes 0)."

**Provenance:** H3 = frozen experiment **E26** (`experiments/e26_independence.py` →
`results/tables/e26_independence.json`, figure `results/figures/e26_matched_stratum.png`), zero-GPU,
23,239 candidates over 5 ensembles × 8 benchmarks (MATH removed 2026-09-01), clustered bootstrap on
config|bench|qid. **Lead H3 with the matched-stratum table/figure, not with AUC.**

---

## RQ spine & Results architecture (REBUILT around H3-new, 2026-09-01)

Center of gravity shifts from *"failure ≠ benefit"* to the **epistemic-provenance principle**. The
failure≠benefit negative is KEPT but as the **motivating bridge** (why we abandon targeting), not the
paper's identity. Arc: **extinction → (targeting fails) → preserve → provenance predicts → INDEPENDENT
ORIGIN MATTERS → conditional utility.**

**RQ spine — RQ → contribution → results section → headline number:**
- **RQ1 — Does debate destroy correct answers?** → YES. (C1, §4) E1/E4a/E18: correct answer
  present-but-forgotten 12.5pp; debate net-destructive on 4/5 benches.
- **RQ2 — Can we recover them by detecting & fixing failing debates?** → detection is easy, *benefit is
  unidentifiable.* (C2, §5–6) hazard ROC 0.73; oracle ceiling +16.5pp; **five** preregistered negatives
  (E13–E17); the conference influence-vote 0.502 *loses* to majority 0.528. ⇒ abandon targeting, act
  structurally. *(This is the bridge, tightened — not the headline.)*
- **RQ3 — Can preserving every hypothesis expose the loss without any targeting?** → YES. (C3, §7)
  E18 candidate recall 0.636 → 0.760, +25pp ceiling, pool ≈ 3.
- **RQ4 — Does a hypothesis's own provenance predict its correctness?** → YES. (C4, §8a)
  E22 P(correct | n_independent) 0.07 → 0.83.
- **RQ5 — (HEADLINE) Is it the ORIGIN of agreement that matters? At matched total support, does
  independently generated support carry more epistemic value than socially inherited support?** → YES,
  near-universal. (C4, §8b) E26 β_ind +0.649 [.565,.737], ~2.68×/vote, 7/8 bench, 5/5 ensembles
  (3B–14B, N∈{3,5}). **Lead with the matched-stratum figure.** Boundary (E27, appendix): *within*
  inherited support, genuine ≈ surface — origin matters, not the reasoning-convergence of later adoption.
- **RQ6 — Can a zero-cost provenance selector turn the principle into accuracy?** → CONDITIONALLY. (C5, §9)
  E23 +3.2pp but regime-bound (16/45 cells); identifiability ∧ availability (E25); the +0.007
  incremental-AUC bridge; E21 external judge fails; E24 convergence handles the constructive regime only.

**Results section flow:** §4 extinction · §5 influence insufficient · §6 predictability ≠ actionability
(bridge) · §7 preservation ceiling · **§8 provenance & epistemic origin (HEADLINE: E22 → E26)** · §9
from mechanism to decision utility (selector, conditional) · §10 generalization consolidation (E25 grid;
note §8's E26/E27 ALREADY span 5 families / 3B–14B / N∈{3,5} / 8 benches) · §11 caveats.

**Appendix / robustness (never main text):**
- **E27 surface-vs-genuine = robustness for §8**: naive event-level P(correct) 0.356 > 0.292 (Δ+0.064) →
  controlled decomposition genuine−surface −0.014 [−.111,+.080] n.s. Present as *one short paragraph* and
  as a "why you must control total support" robustness check for E26 (naive analysis misleads).
- E21 external judge fails; E6 controller (one terse line inside §6); E7/E8/E9/E12; E14–E17 negatives; E20 arith.

**PAPER IDENTITY — DECIDED 2026-09-01 = (A) NEW identity.** Epistemic-provenance / independent-origin is
the headline; failure≠benefit (C2) is the **motivating bridge**, not a co-headline. Title locked above.
Consequences for writing: abstract leads with the extinction phenomenon then pivots fast to the
*"consensus is not epistemically homogeneous"* principle as the central result; the five-negatives
failure≠benefit passage is compressed to the bridge that motivates the structural turn; the closing
reframes reliability around **origin of agreement** (not influence, not benefit-prediction).

---

## Contributions (the paper's spine)

- **C1 — Answer extinction is a real, quantified failure mode.** A correct answer that existed during
  the debate is gone by the final vote; 11.6% of pooled accuracy is behaviorally online-recoverable
  (E1), and the correct answer is *present but forgotten* in 12.5pp of debates (E18 ΔCR). Two
  behavioral mechanisms (E4a: social adoption / self-error drift). *Bonus:* debate is **net-destructive**
  on 4/5 benchmarks — the pre-debate independent answer beats the debated final vote (E19/E23/E24).
- **C2 — Failure predictability ≠ intervention-benefit predictability.** A hazard detector predicts
  extinction (ROC 0.73), and a mechanism-agnostic intervention has a large **oracle** ceiling (+16.5pp,
  E13). But the individual treatment effect is **unidentifiable** from the pre-intervention state —
  cheap features, residualized semantics, an active self-diagnosis probe, a placebo-controlled
  dose-response probe, and even the mid-debate trajectory all fail a preregistered gate
  (**five negatives**, E13–E17). The wall is *sign*, not power (MDE: only ~14% capture needed).
- **C3 — Hypothesis preservation exposes a large, cheap headroom.** Non-destructively preserving every
  hypothesis (H_t ⊆ H_{t+1}) raises candidate recall from 0.636 to 0.760 pooled (ΔCR +0.125, significant
  on all 5 benchmarks; pool ≈ 3), with total headroom over the deployed majority of +25pp (E18).
- **C4 — Consensus is not epistemically homogeneous (the headline mechanism).** A hypothesis's own
  provenance predicts its correctness (independent agreement → 83% correct, debate-emerged → 7%; E22).
  Sharpening this: **at matched total support, independently generated agreement carries more
  correctness signal than socially inherited agreement** — controlling for total consensus, the partial
  effect of independent support is +0.649 [+0.565,+0.737], ~2.68× per vote, holding on 7/8 benchmarks
  and 5/5 ensembles including a 14B anchor and N=5 (E26). This is the near-universal, model-agnostic
  principle and the strongest journal-extension claim; it explains *why* provenance (C-H2) works and
  reframes reliability around the **origin of agreement** rather than influence (bridges to C5's limits).
- **C5 — From mechanism to decision utility: a zero-cost provenance selector, but only in some regimes.**
  A frozen provenance selector (Preserve → Track Provenance → Select) beats the deployed final-majority
  (+3.2pp pooled, up to +8pp) **and** the conference influence-voting (which itself *loses* to majority),
  with no new LLM calls (E23). But the accuracy win is **not universal**: significant in 16/45
  generalization cells, bounded by two applicability conditions — **identifiability ∧ availability**
  (E25) — and the incremental ranking power of origin over raw consensus is only +0.007 AUC
  (collinearity), which is exactly why reranking helps only where headroom remains. An external LLM
  judge does **not** work — a capable judge prefers solving to selecting from a weak pool (E21) — and a
  convergence-gated variant handles the one constructive-debate benchmark but cannot beat round-0 pooled
  because debate is mostly destructive (E24). No fail cell shows a *significant* loss. The residual
  ~86% of the ceiling is the honest open problem. **This was the old "H3"; it is now an application of
  C4, not a headline hypothesis.**

---

## Section map (reuse vs new)

| § | Title | Source / status | Key content |
|---|---|---|---|
| 1 | Introduction | rewrite `introduction_related_work.md` | pivot framing: predict-vs-prevent; failure≠benefit; preservation; **add applicability-conditions to the arc** |
| 2 | Related work | `introduction_related_work.md` | MAD 2026 (iMAD, selective debate), verification/selection, uplift/CATE |
| 3 | Setup | `results_analysis.tex` head + `method.md` | 3 peer 3B LLMs, ≤3 rounds, 5 benches × 5 seeds, robust scoring |
| 4 | **Extinction** (C1) | REUSE RQ4 + E1/E4a + **E18 recall/degradation** | debate net-destroys; 11.6% headroom; two mechanisms; net-destructive 4/5 — **the opening problem** |
| 5 | **Influence is insufficient** | REPURPOSE conference RQ1–RQ3 + E19/E23 influence row | influence concentrated/stable/persona-driven is the natural lever to recover extinct answers — **but influence-voting *loses* to majority (0.502 < 0.528)**; bridges C1→C2 |
| 6 | **Predictability ≠ actionability** (C2) | REUSE `E13_benefit.tex` + **E16/E17 rows** + demoted E6 controller | hazard works; oracle +16.5pp; ITE unidentifiable; 5 preregistered negatives |
| 7 | **Preservation ceiling** (C3) | **NEW `HPMAD_results.tex` §A** | E18 candidate recall final vs history; ΔCR; pool size; +25pp headroom |
| 8 | **Provenance & epistemic origin** (C4 — HEADLINE) | **NEW `HPMAD_results.tex` §B** + `e26_independence` | E22 provenance→correctness signal; **E26 independent > inherited at matched total support** (β_ind +0.649, 2.68×/vote, 7/8 bench, 5/5 ens incl 14B & N=5) — *lead with the matched-stratum table/figure*; H1/H2/H3 generalization grid (E25: H1 15/15, H2 14/15) |
| 9 | **From mechanism to decision utility** (C5 — demoted old H3) | **NEW `HPMAD_results.tex` §C** | E23 selector beats deployed majority (+3.2pp) & conference influence-voting, zero cost, **but regime-bound (16/45 cells)**; applicability conditions **identifiability ∧ availability** (E25 H3 7/9 reasoning / 0/6 degenerate); the +0.007 incremental-AUC bridge; E21 external judge fails; E24 convergence; *no fail cell a significant loss* |
| 10 | Robustness / honest caveats | REUSE `E6_robustness.tex` caveats + new | cluster-bootstrap, per-bench, capability-mismatch, offline-replay, **N>3 remaining** |
| 11 | Conclusion + future work | new | Preserve→Verify→Select; **N>3 + live HP-MAD** are the remaining passes |

**Repurpose / demote:**
- The **conference influence layer** (RQ1–RQ3) is no longer inert "background" — it becomes §5
  *"Influence is insufficient"*: the natural first hypothesis for recovering extinct answers, falsified by
  influence-voting losing to majority. Frame with purpose, not as post-hoc description.
- The E6 hazard-guided **controller** (E6–E12) moves from "hero" to a *supporting* result inside §6/C2
  (matches always-on at ~half the interventions but detector edge over random ≈0 — motivates "targeting
  is the hard part"). Do not lead with it.

---

## Selector results table (C5 downstream utility — build as `tab:hpmad`)

*(The paper's HEADLINE table/figure is the E26 matched-stratum result in §8, not this selector ladder;
this table supports the demoted C5 "decision utility" subsection.)*

Accuracy of each *inference-time selection rule* over the frozen debate logs (5 bench × 5 seed,
gold-free selection, correctness by the robust scorer). Cluster-bootstrap 95% CI over questions.

| Selector | pooled | gsm8k | bbh | strategyqa | csqa | tqa | extra LLM calls |
|---|---|---|---|---|---|---|---|
| Final majority (deployed) | 0.528 | 0.760 | 0.466 | 0.604 | 0.664 | 0.148 | 0 |
| Conference influence-voting | 0.502 ↓ | 0.706 | 0.430 | 0.602 | 0.622 | 0.150 | 0 |
| Round-0 (independent) majority | 0.560 | 0.810 | 0.494 | 0.580 | 0.750 | 0.166 | 0 |
| **Provenance selector (ours)** | **0.564** | 0.808 | 0.502 | 0.590 | 0.750 | 0.174 | **0** |
| Oracle (history pool ceiling) | 0.760 | 0.920 | 0.798 | 0.908 | 0.936 | 0.240 | — |

Reading: preservation ceiling is +23pp over deployed; a zero-cost provenance rule captures ~14% of it
and beats both the deployed majority and the conference method.
*External LLM judge lives in prose only* (E21 was a 4-bench pilot, final-maj 0.595 → J(history) 0.527 =
−0.068; do NOT put it in this 5-bench table — different sample. Its point is qualitative: it loses, and
the 14B judge solves better (0.76) than it selects (0.53)).

---

## Figures to produce

1. **Extinction schematic + rate** (C1): the 42→48 worked example + per-bench ΔCR bar (E18).
2. **Failure≠Benefit** (C2): oracle ceiling vs learned-CATE ≈ random, across the 5 preregistered
   attempts (E13–E17) — one panel per attempt, effect-AUC ≈ 0.5 line.
3. **Provenance signal** (C4): P(correct | n_independent = 0,1,2,3) climbing 0.07→0.83, per bench (E22).
4. **HEADLINE — matched-stratum** (C4/H3, E26, *already produced* `results/figures/e26_matched_stratum.png`):
   P(correct) vs n_independent, one line per total-support stratum — at fixed total support, independent
   origin raises correctness (sup=4: .21→.44→.66→.74). This is the lead H3 evidence, not AUC.
5. **Selector ladder** (C5): the master table as a grouped bar (final-maj / influence / round0 /
   provenance / oracle) pooled + per bench, with the net-destructive annotation on 4/5.

---

## Abstract (draft)

> Multi-agent debate (MAD) among LLMs is widely used to improve answer reliability, but we show it
> routinely **destroys** correct answers: on four of five benchmarks a correct hypothesis that existed
> during the debate is gone by the final vote, and the pre-debate *independent* answer beats the debated
> majority. A natural fix is to detect and intervene on failing debates. We show the detection half is
> easy — a hazard detector predicts extinction (ROC 0.73) and a mechanism-agnostic intervention has a
> large oracle ceiling (+16.5pp) — but the **targeting** half is not: *which* debate benefits from
> intervention is unidentifiable from the pre-intervention state — across five preregistered attempts a
> learned benefit predictor never beats random despite the large oracle ceiling (a failure of *sign*,
> not statistical power). We therefore abandon targeting and address extinction *structurally*:
> **Hypothesis-Preserving debate** never
> destructively overwrites a hypothesis and re-selects the final answer from the full history pool,
> which raises candidate recall from 0.64 to 0.76. We then identify *why* provenance helps: **consensus
> is not epistemically homogeneous** — at matched total support, independently generated agreement
> carries substantially more correctness signal than socially inherited agreement (partial log-odds
> +0.65, ~2.7× per vote, on 7 of 8 benchmarks and 5 ensembles spanning 3B–14B and N∈{3,5}). Realizing
> this as a decision rule with an external LLM judge fails — a capable judge does better solving the
> question than selecting from a weak pool — but a zero-cost **provenance-aware** selector, which trusts
> independently and persistently supported hypotheses over those that only emerge through social copying,
> recovers part of the headroom: it beats the deployed majority (+3.2pp, up to +8pp) and the conference
> influence-voting baseline with no additional inference. This selector gain is **conditional** (origin
> adds little to raw consensus ranking once they co-vary), and we characterize its two applicability
> conditions — provenance must be *informative* about correctness, and a correct hypothesis must be
> *available* in the pool —
> showing they hold across debater capability and model family on multi-answer reasoning benchmarks,
> while binary (StrategyQA) and adversarial (TruthfulQA) benchmarks each violate one. Our central finding
> is that **consensus in multi-agent debate is not epistemically homogeneous** — the *origin* of agreement,
> not its magnitude nor the models' mutual influence, is what tracks correctness — reframing reliability
> around preservation and provenance and quantifying a large, still-open headroom.

---

## Honest caveats to keep (do not hide)

- **H3 is observational, not a manipulation.** E26 is a logistic on candidate features; the matched-
  stratum contrast is causal-flavored (conditions on total support) but still observational, and
  n_independent partly encodes question difficulty (E22). Claim "carries more correctness *signal* at
  matched total support", NOT "is more *valuable*/causal". Report the difficulty association as a
  limitation — and note it is part of the point (independent agreement marks better-determined
  questions = wisdom-of-independent-crowds).
- **H3's mechanism is real but its incremental ranking power is small.** Origin adds only +0.007 AUC
  over raw consensus (0.83 collinear). This is NOT a weakness to hide — it is the quantitative reason
  the C5 selector is regime-bound; put it beside H3 as the mechanism→utility bridge.
- **Downstream selector (old H3 → C5) is modest** (~14% of ceiling, ties round-0-majority; significant
  in 16/45 cells). Never say it "degrades gracefully / doesn't hurt" — say **"no fail cell shows a
  significant loss (every CI includes 0)."** The sophistication is not what wins — independent agreement is.
- **Generalization is broad but N and persona⊥backbone remain partial.** E25 now spans **debater
  capability + model family (5 families, 3B–14B) AND N∈{3,5}** over **8 benchmarks** (MATH removed
  2026-09-01, Step-A parse-fail): H1 extinction, H2 provenance→correctness, and **H3 independent>inherited
  (7/8 bench, 5/5 ens incl E_strong 14B & E_n5 N=5)** all hold; the selector gain stays regime-bound.
  **N>5 and a fully crossed persona×backbone remain the open strengthening pass**, with live (non-offline)
  HP-MAD. Do NOT claim generalization of CATE/controller.
- Gains are **benchmark-regime-dependent** (debate net-destructive on 4/5; StrategyQA constructive).
- All selection numbers are **offline replays** of a frozen destructive-protocol debate; end-to-end
  HP-MAD (a live non-destructive protocol) is future work, motivated but not run.
- **Unify the majority baseline before camera-ready.** E18 reports final-majority 0.509 (mode,
  arbitrary tie-break) while E19/E23 report 0.528 (deterministic tie-break); ~2pp gap, same quantity.
  Recompute all tables against the E23 deterministic majority so tab:preserve and tab:selector agree.
