# E25 breadth extension — 4 new benchmarks — PRE-REGISTRATION (locked 2026-08-31)

**Written BEFORE any result is seen.** Purpose: run the **frozen** E25 protocol (E18/E22/E23 verdicts,
method frozen at E23 — no selector tuning, no feature changes, no gate changes) on four held-out
benchmarks, as an **out-of-sample prediction test of the §9 applicability-conditions framework**
(identifiability ∧ availability). This is NOT an attempt to find more H3-pass cells and NOT a
benchmark-shopping exercise — the disposition of *every* benchmark that is run is fixed by the rules
below, decided in advance.

Benchmarks: **AQuA-RAT** (5-way MC), **MATH** (open numeric), **ARC-Challenge** (4-way MC),
**MMLU** (4-way MC, knowledge).

---

## 1. Frozen hypotheses & gates (identical to E25, verbatim)

Per cell = (ensemble, benchmark, seed). Ensembles = E_orig (3B), E_7b (7–8B), E_mixed (9B+small).

- **H1 — extinction generalizes:** ΔCR = CR_history − CR_final > 0. Gate: `dCR_ci[0] > 0` (cluster-boot qid).
- **H2 — provenance → correctness:** `n_independent` dirAUC > chance. Gate: `dir_ci[0] > 0.5` (+ monotonicity reported).
- **H3 — provenance selector still helps:** selector Acc > final-majority. Gate: `prov_equalweight.ci[0] > 0 OR prov_logistic_xfit.ci[0] > 0`.

## 2. Pre-registered predictions from §9 (recorded now, before results)

The applicability-conditions framework predicts these benchmarks are **H3-difficult**, each for a
specific, named reason. Recording the prediction is the whole point of the test.

| bench | regime | identifiability (prov informative?) | availability (correct in pool?) | **H3 prediction** | H1/H2 prediction |
|---|---|---|---|---|---|
| MATH | open numeric, "sticky" answers (prior extinction yield ≈2%) | plausible | **LOW** (few forgotten-correct) | **FAIL** (availability) | H1 may be weak/absent; H2 plausible |
| AQuA-RAT | 5-way MC | **LOW-MID** (small closed set) | mid | **FAIL** | H2 likely weak (MC) |
| ARC-Challenge | 4-way MC | **LOW-MID** | mid | **FAIL** | H2 likely weak |
| MMLU | 4-way MC, factual recall | **UNCERTAIN** (knowledge, not reasoning) | mid | **FAIL** (lean) | H2 uncertain |

A confirmed prediction (these fail H3 for the named reason, while H1/H2 hold where the regime is valid)
**strengthens §9** — it turns the post-hoc characterization into a predictive theory. A surprise (H3
passes on a closed-MC bench) is a *finding*, reported as such, that refines the framework.

## 3. Pre-committed disposition rule (objective — no post-hoc benchmark selection)

After the pilot (all runnable ensembles × seed42 × n=100), classify each benchmark by applying these
gates to the pre-registered metrics. **Every benchmark run appears somewhere** — main table, §9 boundary,
appendix, or an excluded-with-criterion table. Nothing is silently dropped.

**Step A — Validity gate (is the pipeline/regime valid at all?).** A benchmark is VALID iff ALL hold
(pooled across ensembles):
1. **Parse validity:** censored/parse-fail rate ≤ 15% (answer_extract not mostly CENSORED).
2. **Non-degenerate accuracy:** 0.10 ≤ final-majority acc ≤ 0.95 (not floor/ceiling-trivial).
3. **Availability floor (phenomenon present):** CR_history ≥ 0.15 — there must exist correct hypotheses
   in the pool for preservation/selection to be *testable* at all.

A benchmark FAILING step A → **EXCLUDED from the main claims**, listed in an "excluded benchmarks" table
**with the specific failed criterion and its number** (e.g. "MATH excluded: CR_history 0.08 < 0.15,
availability floor — the extinction phenomenon does not occur, so H1/H3 are undefined here"). This is a
stated exclusion criterion, not a quiet removal.

**Step B — Among VALID benchmarks, disposition by hypothesis signs** (H_k "holds" = gate passes in ≥2/3
ensembles, the E25 reporting convention):
- **H1 ∧ H2 ∧ H3 hold** → **MAIN** generalization table (adds a genuine H3-pass generalization cell).
- **H1 ∧ H2 hold, H3 fails** → **§9 BOUNDARY-CONDITION** benchmark: an out-of-sample confirmation of the
  applicability conditions (provenance signal present but availability/ceiling limits realized gain).
  Reported in §9, NOT dropped.
- **H1 or H2 also fails (valid pipeline)** → **APPENDIX** characterized negative, with the regime reason.

## 4. Scope & escalation (GPU discipline)

- **Pilot:** n=100, seed42, 3 ensembles. For new benches E_orig has no frozen logs (original run covered
  only gsm8k/strategyqa/csqa/bbh/truthfulqa + aqua_seed42), so E_orig new-bench cells are **generated**
  into the tagged `data/debate_logs/gen/` path using the frozen 3B persona/backbone spec (infrastructure,
  method still frozen; frozen originals in `data/debate_logs/*.jsonl` are never touched). `aqua/E_orig/seed42`
  is reused where present.
- **Escalation (seeds 43-45, n→400):** ONLY for a benchmark that passes Step A **and** shows H1∧H2 signal
  — identical to the E25 Stage-2 rule. Do **NOT** expand a benchmark just because H3 is a near-miss
  (anti-chasing discipline, consistent with E24/E21 STOP rules).
- **GPU:** generation → gpu1-pinned Ollama `:11435` ONLY, gpu0 untouched, never disturb the other user's
  PID ([[gpu-run-pinning]]). Verdict is zero-GPU. Linux python `/home/alex/venvs/env/bin/python`.

## 5. New engineering required (NOT method changes)

- AQuA: data ✅ (`data/processed/aqua.jsonl`), scoring ✅ (MC letter) → generate E_7b/E_mixed (+E_orig).
- MATH: data ✅, scoring ✅ (numeric) → generate all 3.
- MMLU: scoring ✅ (MC letter) but **data missing** → download + prepare `data/processed/mmlu.jsonl`.
- ARC-Challenge: **fully unwired** → download + prepare + add scoring + answer_extract entries (MC letter).
- Extend the E25 bench list / call `verdict(benches=[...])` to include the four (bare `--verdict` only
  covers PILOT_BENCHES — same gotcha noted in the E25 Stage-2 memory).

**Anti-cherry-pick commitment:** the four benchmarks, their validity numbers, and their dispositions per
§3 will all be reported. Good → main; valid-but-negative → §9 boundary/appendix; invalid → excluded with
criterion. No benchmark is removed after seeing its H3 result without a §3-Step-A reason.

---

## ADDENDUM (locked 2026-08-31, BEFORE any result) — model-scale + N-scaling conditions

Written before generation. This EXTENDS the design along two new axes without changing the frozen
method or the H1/H2/H3 gates (identical to §1). Two new ensemble conditions are added and run across
the **full 9-benchmark set** (5 original + AQuA + ARC + MMLU + MATH) as **main-table generalization
cells** (user decision 2026-08-31):

- **E_strong** (strong-lite anchor): `{Qwen2.5-14B (careful), Phi4-mini (skeptic), Llama3.2-3B (bigpicture)}`.
  Fits one 16 GB card (9.0+2.5+2.0 = 13.5 GB, no model swapping). Deliberately parallels **E_mixed**
  (Gemma2-9B → Qwen2.5-14B, same Phi + Llama3B) so the contrast **isolates the strong-anchor effect**.
  Answers: *does the phenomenon survive when the team's lead model is materially stronger (14B)?*
- **E_n5** (N=5 structural stress test): `{Qwen2.5-3B, Phi4-mini, Llama3.2-3B, Mistral-7B, Qwen2.5-7B}`
  = E_orig's three agents + two mid agents (unique agent names, personas reused from the canonical 3).
  Fits 16 GB (15.5 GB). Answers: *do H1/H2/H3 hold when the debate has 5 agents instead of 3?* This is a
  structural robustness axis, **not** a new experimental factor to Cartesian-expand.

**Frozen-method guarantee for N=5:** verified zero-GPU (mock N=5 debates → the SAME e18/e22/e23
extractors process 5 agents; `n_independent` scales to the agent count; agent names unique so
`run_debate` does not collapse them). No verdict-code change; only `ENSEMBLES` + `_agents_spec`
(now accepts an optional 4th persona element) changed.

**Ensemble ladder (final):** small (E_orig) → mid/cross-family (E_7b) → mixed-capability (E_mixed) →
strong-anchor (E_strong); plus N=3→N=5 (E_n5). ~7 checkpoints across 5 families (Qwen/Phi/Llama/
Mistral/Gemma), ≈3B–14B.

**Scope & GPU discipline (this addendum):** E_strong + E_n5 pilot = 2 ensembles × 9 benches × seed42
× n=100 = 18 cells. Escalation seeds 43-44 (n→300) ONLY for cells passing Step-A validity **and**
H1∧H2 (same anti-chasing rule as §4; H3 near-miss is NOT a reason to expand). **These runs go on
gpu0** via a dedicated gpu0-pinned Ollama server (`:11436`, `CUDA_VISIBLE_DEVICES=0`, `OLLAMA_VULKAN=0`);
the concurrently-running new-bench pilot on **gpu1 (`:11435`, PID must not be disturbed)** is left
untouched. If the other user reclaims gpu0, yield it — never preempt ([[gpu-run-pinning]]). Disposition
of every E_strong/E_n5 cell follows §3 verbatim; nothing dropped after seeing H3.
