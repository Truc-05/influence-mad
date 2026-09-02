# E21 — Semantic selector over the frozen hypothesis pool (PREREG, LOCKED 2026-08-28)

## Motivation (positive-evidence ladder, NOT a CATE/controller rescue)
E18 PASS (preservation ceiling: pooled candidate recall 0.636→0.760, ΔCR +0.125, pool ≈3 hyp) →
E19 PASS (pool is *selectable*: every gold-free heuristic beats final-majority, +2.5–3.2pp, but
captures only ~12%; deterministic arithmetic verify E20 = negative, errors are semantic not
computational). ⇒ the remaining ceiling needs a **semantic selector**. E21 tests it on the
**frozen** history pool (no HP-MAD regeneration yet).

## What the judge may and may NOT do
Judge input = question + the candidate hypotheses **with their reasoning/provenance**. The judge
must **select one candidate from the pool**, never generate a new answer:
$$\hat h = \arg\max_{h\in\mathcal H} V(q, h, r_h).$$
Enforced by a labelled multiple-choice format (`CHOICE: <label>`); a produced answer outside the
pool is a parse-fail, not a new candidate. This blocks "the judge is just a stronger LLM solver."

## Arms (same judge model+budget across all; only the input set changes)
| arm | input | role |
|---|---|---|
| `final_majority` | logs only (no GPU) | deployed baseline |
| `J(final)` | question + FINAL-round distinct candidates + reasoning | **control** (judge, no preservation) |
| `J(history)` | question + ALL-round distinct candidates + reasoning | HP selector |
| `J(q)` | question only, judge solves directly (matched model+budget) | control: is the pool even needed? |

Judge model = **qwen2.5:7b**, temperature 0 (deterministic verification), num_predict 512 (matched
across arms), gpu1 `:11435` ONLY (never :11434/gpu0). Candidates labelled in deterministic order
(first-seen round, then answer string); representative reasoning = longest supporting reasoning,
truncated; provenance = first-seen round. Support COUNT is withheld (so the judge verifies, does
not re-vote). Gold is used ONLY to score correctness after selection (bucket→correct via the same
`e3.prep_log`/`is_correct` as every prior experiment).

## Primary gate (isolates preservation from "a stronger judge")
$$\Delta_{HP} = Acc[J(\mathcal H_{history})] - Acc[J(\mathcal H_{final})] > 0,$$
cluster-bootstrap 95% CI over qid, PASS iff lo > 0. Same judge, same prompt template — the ONLY
difference is whether extinct hypotheses are in the candidate set.

## Secondary (report-only, non-veto)
- $Acc[J(\mathcal H_{history})] - Acc[\text{final majority}]$
- ceiling capture $= \dfrac{Acc[J(\mathcal H_{history})] - Acc[\text{final maj}]}{CR_{history} - Acc[\text{final maj}]}$
- **pool-value control**: $Acc[J(\mathcal H_{history})] - Acc[J(q)] > 0$ (pool adds value over the
  judge solving alone at matched model/budget — defeats "you just added a stronger LLM").
- resurrection usage: on H_history≠H_final debates, fraction where J(history) picks a candidate
  absent from H_final, and its accuracy (proves the gain comes from recovered historical candidates).
- per-arm parse-fail rate; per-benchmark breakdown.

## Two-stage execution (do NOT burn 2500 calls blind)
**PILOT** — stratified ~400 debates (100 each: GSM8K/BBH/StrategyQA/CSQA), resumable cache.
Directional PILOT gate (all must hold):
1. sign(Δ_HP) > 0;
2. Acc[J(history)] > Acc[final majority];
3. no benchmark with catastrophic degradation (J(history) < final_maj − 3pp on any bench);
4. selector actually uses recovered historical candidates (resurrection rate > 0 and net-positive).

PILOT negative → **STOP**, do not spend the full run. PILOT clearly positive → **freeze the prompt**,
then run **FULL 2500** (5 bench × 5 seed) for the powered primary gate.

## If FULL passes
Proceed to end-to-end **HP-MAD**: Preserve → Verify → Select, under a non-destructive debate
protocol + model-family generalization. The method is then a clean positive:
$$\boxed{\text{Preserve} \rightarrow \text{Verify} \rightarrow \text{Select}}.$$
