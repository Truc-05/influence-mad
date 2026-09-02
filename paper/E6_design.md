# E1→E5 synthesis + E6 design (FailSafe-MAD)

*Frozen 2026-08-26, after E5 passed its GPU gate. Numbers from `results/tables/*.json`.*

---

## 1. Claim → Evidence → Gate → Implication (E1–E5)

| Exp | Claim | Evidence (frozen, robust re-scoring) | Gate | Verdict | Feeds into E6 |
|---|---|---|---|---|---|
| **E1** | Debate net-destroys correct answers; some loss is recoverable *only* online, not by any aggregator. | Pooled-4 baseline **60.7%**; **behavioral online-recoverable = 11.6%** (gsm8k 7.4 / sqa 11.0 / csqa 12.0 / bbh 15.8); vote-recoverable 16.2%; parser-censored ≈0.9%. | Headroom ≫ parser noise. | **PASS** | Sets the ceiling worth chasing (~11.6 pp) and the unit of analysis: the *extinction event*. |
| **E4a** | Extinction has ≥2 behavioral modes; the dying holder is usually a lone minority that herds. | Pooled-4: death-by-adoption **0.72**, dtype social **0.72** / self **0.25** / parsefail 0.03, single-holder **0.78**, minority-detectable-no-gold **0.43**, t\* split 46%/54% (round 1/2). GSM8K clean anchor: self 21 ≥ social 16. | Is there exploitable structure? | **PASS** | The treatment must protect a lone correct minority from herding → motivates the *combined* bundle. |
| **E2** | Conventional influence analytics identify *which* mode. | Influence-only AUC **0.49** ≈ perm floor 0.51; structural 0.63; infl+struct 0.62; GSM8K `infl_gap` MWU p=0.72. | STRONG-GO for an influence-based router. | **FAIL** → demote PageRank | Don't route by predicted mechanism → use a **mechanism-agnostic** treatment. |
| **E3a** | Extinction is predictable *early* from frozen transcript dynamics. | n=2952, prev 12.1%; **ALL\|lr ROC 0.757**, AUPRC 0.244 (lift 2.02); F1+F2+F3\|hgb AUPRC 0.249 (lift 2.07). **LOBO ROC 0.67–0.75** on every held-out bench incl. open-answer GSM8K. F3 adds only ~0.03 AUPRC. | LOBO generalizes. | **PASS — HERO** | **The E6 trigger.** But no high-precision operating point (R@P.5≈0, P@R.5≈0.25) → soft τ, budget for FPs. |
| **E3b** | We can route social-vs-self online. | LR/HGB unstable 0.55↔0.78; struct(F1)\|hgb ROC 0.78 but self-recall 0.33; **LOBO→GSM8K ROC 0.57 ≈ chance**. | LOBO > 0.65. | **FAIL** → secondary | Confirms E2: no mechanism gating → single treatment. |
| **E5** | Intervening at the trigger with the combined bundle actually saves the answer under clean re-query. | n=116 (gsm8k 37 + bbh 79). control preserve **50%** / rec_maj **23%**. **combined preserve 66% (Δ+16 pooled, +12 macro), rec_maj 30% (Δ+7)**; self_verify 59%(+9)/+0; influence_resistant 56%(+6)/+1. BBH combined preserve **71% (Δ+22)**. Calls: combined 4.32 ≈ control 4.37 (**no extra call cost**). | Combined preservation/recovery clearly > control re-query. | **PASS** | Treatment works and is ~free per event → E6 is now well-posed. |

**One-line arc:** *There is recoverable loss (E1), it is a lone-minority herding failure (E4a) that influence can't diagnose (E2/E3b) but transcript dynamics can predict early (E3a), and a mechanism-agnostic combined re-query at the trigger recovers it (E5).* → **E6: is predicting *where* to intervene worth it?**

---

## 2. E6 — hazard-triggered adaptive controller

### 2.1 Objective (target inequalities)

Accuracy:  `never-intervene  <  hazard-triggered(τ)  ≈  always-combined`

Cost:      `cost_triggered  ≪  cost_always`   (intervene on only ~20–40% of turns)

Detector value:  `hazard-triggered(τ)  >  random-trigger @ matched budget`  ← the experiment that justifies E3a existing.

### 2.2 The key methodological move — precompute both counterfactuals, then all policies are offline

The expensive part (GPU re-query) is decoupled from the cheap part (policy sweep):

1. Take a debate set **D** (gsm8k + bbh logs; the same two regimes as E5). For each debate pick a **decision gate** round `r*` (see 2.4).
2. Roll forward **twice** from the logged state at `r*` under the E5 machinery (`run_arm`): once `control` (plain re-query), once `combined`. Record per debate: final maj-correct under each arm, and cost (agent-turns).
3. Compute the E3a **hazard score** at `r*` from frozen `rounds[0..r*]` features (reuse `e3_router._features` + a Stage-A model trained LOBO, so the debate being scored is never in the detector's training fold).

Now every controller is a **selection rule over the two precomputed outcomes** — zero extra GPU:
- **never** → always adopt `control` outcome. (accuracy floor, 0% intervention)
- **always** → always adopt `combined` outcome. (accuracy ceiling?, 100% intervention)
- **hazard(τ)** → if `hazard ≥ τ` adopt `combined` (pay combined cost) else `control`.
- **random(p)** → adopt `combined` on a random p-fraction, sized to match hazard(τ)'s intervention rate.

Sweep `τ∈[0,1]` → the **accuracy vs intervention-rate curve** (the README's E6 hero figure). Overlay the random curve (averaged over many draws — analytic, free). Bootstrap over debates for CIs.

*This makes E6 cheap and rigorous: one fixed GPU cost (2 rollouts/debate), everything else is CPU post-processing, and the τ-sweep / random-baseline / bootstrap are exact rather than sampled.*

### 2.3 What "cost" means — and the sharper accuracy story (important)

E5 shows the combined clause costs ~0 extra **calls**. So E6 must define cost as the two things that are actually spent:
- **intervention rate** = fraction of agent-turns given the heavier prompt (≈ added reasoning tokens + latency). This is the README's axis.
- **collateral accuracy risk on would-live debates** — *this is the strongest reason to target.* The influence-resistant / "demand a concrete error" clause makes agents refuse to move; applied to a *healthy* debate it can suppress **beneficial revisions (revivals)** — E1 counts 141 pooled revivals (gsm8k 23, sqa 41, csqa 14, bbh 47). So **always-combined may *lower* accuracy on the 88% of debates that were fine**, not just cost more tokens.

→ Measure, per arm, on the partition {would-die: control outcome wrong} vs {would-live: control outcome correct}:
  - help rate = P(combined flips wrong→correct | would-die),
  - harm rate = P(combined flips correct→wrong | would-live).
  A targeted controller wins if it concentrates intervention on would-die and spares would-live. If always-combined's net effect is negative on would-live, then **hazard(τ) can beat always-combined on accuracy *and* cost** — the strongest possible E6 result.

### 2.4 Decision-gate choice (the one real design tension)

Deaths split 46%/54% across t\*=1 / t\*=2. A death at t\*=1 is already gone once round-1 revisions are visible, so:
- **Gate at r\*=0** (predict from initial answers, intervene on round-1 revision onward): catches *all* deaths, but the detector at r\*=0 has no F2 dynamics yet (structure/text only) → weaker ranking.
- **Gate at r\*=1** (F2 dynamics available → stronger detector): misses the ~46% that die at t\*=1.

**Recommendation:** v1 = **two-gate** controller (score at r\*=0 *and* r\*=1; intervene at the first gate whose hazard ≥ τ). It needs at most rollouts from r\*=0 (which already cover the round-1..F window), so cost is unchanged vs a single r\*=0 gate; the r\*=1 gate is just a second, stronger scoring opportunity for the debates not yet triggered. Report single-gate r\*=0 as an ablation.

### 2.5 Arms, metrics, and the go/no-go

| Arm | Policy | Purpose |
|---|---|---|
| never | adopt control | accuracy floor / cost 0 |
| always-combined | adopt combined ∀ | accuracy ceiling & full cost |
| **hazard(τ)** | combined iff hazard≥τ | the proposed controller (sweep τ) |
| random(p) | combined on random p% | matched-budget detector-value control |

Primary metric: **final maj-accuracy vs intervention-rate** curve. Secondary: help/harm decomposition (2.3), tokens, calls.

**E6 PASS if:** at ~20–40% intervention rate, hazard(τ) keeps **≥ ~70% of the always-combined accuracy gain** *and* its curve lies **significantly above random** (bootstrap CI on the accuracy gap / area between curves excludes 0). **Clean-negative if** hazard ≈ random (detector adds nothing) or always-combined ≤ never (treatment doesn't generalize off the extinction subset) — both are reportable.

### 2.6 Cost / feasibility (single idle GPU, via Ollama)

D = gsm8k + bbh, ~500 debates × 2 arms × ~2 forward rounds × 3 agents ≈ **6k calls** (~2.5–5 h on one 3B-model GPU). Sample down or gate at r\*=1 to halve it. Reuses `e5_counterfactual.run_arm` + `e3_router._features`; new code is the gate loop, the LOBO Stage-A scorer, and the offline policy/curve/bootstrap.

**Runtime constraint:** run on the **idle GPU only** — start the Ollama server with `CUDA_VISIBLE_DEVICES=<idle>` (pinning the python client is not enough; the debate calls execute inside the Ollama server). Never disturb the other GPU's job.

### 2.7 After E6

Only if E6 passes and a top venue needs more generalization: E7 (semantic/embedding Stage-B upgrade) and E8 (N>3 agents, more backbones/tasks, topologies). Not before — E6 decides whether the paper is "a framework" or "an analysis + intervention".
