# Figure plan — FINALIZED & de-duplicated (2026-09-01)

Locked against the H3-new architecture (`PAPER_OUTLINE.md`). Rule enforced: **every quantity has ONE
home**; per-result figures show pooled/primary evidence, the generalization *atlas* (Fig 6) owns the
full config×bench grids, the *utility* figure (Fig 5) owns everything selector/ΔAcc. Design rule per
figure: **20–30% conceptual visual thesis + 70–80% quantitative evidence**, one headline per figure.

## Corrected denominators (MATH removed → 8 benchmarks; use EVERYWHERE, do NOT write 45)
Cells = (config × benchmark) across all verdicts: Stage-1 **15** (E_orig/E_7b/E_mixed × 5 core) +
breadth **9** (× aqua/mmlu/arc) + scale **16** (E_strong/E_n5 × 8) = **40 cells**.
- **H1 extinction:** 40/40 directional (ΔCR>0), **35/40** strict (CI-lo>0; 5 near-ceiling near-misses).
- **H2 provenance→correctness:** **39/40** (only E_7b/StrategyQA fails).
- **H3-OLD selector > majority:** **15/40** (was mis-stated 16/45; recompute killed E_7b/math).
- **H3-NEW independent>inherited (E26):** 7/8 benchmarks, 5/5 ensembles (pooled marginals).

## Visual language (fixed across the whole paper)
- **Independent support = deep blue** · **socially inherited = orange** · correct = green (only when
  encoding correctness) · extinction/harm = red · preserved/history = teal · baseline/majority = grey.
- No rainbow/jet. White bg, faint grid, no top/right spines, one sans family, thin whisker/ribbon CIs,
  large A/B/C labels same position. **Hypothesis graphs colored by HYPOTHESIS identity, not agent.**

---

## MAIN FIGURES (7; fold Fig 7 → Intro to reach 6)

Legend for readiness: ✅ = drawable now from a frozen JSON/log · ♻ = needs a *cheap zero-GPU recompute*
from the candidate table (no new runs) · ✎ = conceptual/schematic (design + pick a representative log).

### Fig 1 — HP-MAD in one picture (method) · 3 panels · [Setup, not Results]
Thesis: *Preserve → Track Provenance → Select, zero extra inference.*
- A ✎ pipeline schematic (destructive MAD vs hypothesis-preserving; H_t ⊆ H_{t+1}).
- B ✎ provenance features cartoon (n_independent / n_persist / first_round on one hypothesis).
- C ✎ selection step (pick-one from the preserved pool).

### Fig 2 — Correct hypotheses are forgotten (H1) · 4 panels
Thesis: *the right answer was present, then socially erased.* (E18; E1/E4a)
- A ✎ hero trajectory of ONE real debate (correct answer alive @round0 → extinct by final); below it
  `final pool {B}` vs `history pool {A,B,C}`. *(pick the canonical GSM8K 42→48 log.)*
- B ✅ dumbbell CR_final ● → CR_history ● per benchmark (gap = forgotten-correct). `e18…json`
- C ✅ ΔCR forest + bootstrap CI, per benchmark + pooled. `e18…json`
- D ✅ pool-size expansion: |H_final| vs |H_history| distribution (≈1.7 vs ≈3.1). `e18…json`
  *(Generalization of H1 (bench×config) and scale → moved to Fig 6; NOT here.)*

### Fig 3 — What predicts hypothesis correctness (H2) · 6 panels
Thesis: *a hypothesis's own history is a correctness fingerprint.* (E22)
- A ♻ **fingerprint**: mean provenance profile of correct vs wrong hypotheses (indep/support/persist/
  late-adopt bars). *(recompute means from candidate table)*
- B ✅ feature-AUC forest, all 6 features, vertical ref at AUC=0.5. `e22…json`
- C ♻ n_independent reliability curve 0→3 with Wilson/bootstrap ribbon. `e22 bins` (+CI recompute)
- D ♻ n_persist reliability curve. `e22 bins`
- E ✅ first_round effect (late-emerging → wrong: .37→.08→.04). `e22 bins`
- F ✅ null features: late_adopt / resurrected ≈ chance (honesty panel). `e22 bins`
  *(H2 generalization matrix → Fig 6; NOT here.)*

### ⭐ Fig 4 — NOT ALL CONSENSUS IS EQUAL (H3, HERO, ~full 2-col page) · 8 panels
Thesis (money): **same total support ≠ same reliability; origin is what matters.** (E26)
- A ✎ **visual paradox**: Candidate A (3 independent origins) vs Candidate B (1 origin + 2 inherited),
  `TOTAL SUPPORT = SAME` → `≠ same reliability`. Reviewer gets novelty before any statistics.
- B ✅ **matched-support heatmap** with cell numbers (sup4: .21/.44/.66/.74 …), faint "more independent
  origins →" arrow. `e26 matched_stratum`
- C ✅ matched-support **curves** (one line per total-support stratum, all sloping up). *(already drawn:
  `e26_matched_stratum.png`)* — B and C are the two canonical reads of the signature result (grid vs trend), intentionally complementary.
- D ♻ **controlled-β forest**: β_ind under {ctrl total support} → {+bench FE} → {+config FE} → {full},
  all right of 0. Kills the "just a proxy for total support" objection. `e26 model_A/FE` (+2 intermediate rungs recompute)
- E ✅ **per-vote** independent (1.035) vs inherited (0.386) two-bar, `2.7×` callout (caption: conditional
  association / per-vote coefficient, NOT causal). `e26 model_B`
- F ✅ per-**benchmark** β_ind forest, 8 benches — StrategyQA CI crossing 0 kept (raises trust). `e26 per_benchmark`
- G ✅ per-**config** β_ind forest, 5/5 incl E_strong(14B), E_n5(N=5). `e26 per_config`
- H ✎ takeaway strip: *"Consensus quantity = how much agreement; provenance = how it was formed."*
  → **candidate for the graphical abstract.**

### Fig 5 — From mechanism to decision utility (why modest + when useful) · 8 panels
[MERGES the user's old Fig 5 "why" + Fig 6 "when". Owns ALL selector/ΔAcc content.]
Thesis: *the mechanism is real but a rank-based selector helps only where headroom + identifiability co-occur.*
- A ♻ collinearity scatter n_ind vs n_support, r=0.83 (jitter). `e26 corr` (+scatter recompute)
- B ✅ three-AUC lollipop: independent .811 / support .841 / both .849, bracket **Δ+.007**. `e26 incremental_auc`
- C ✎ contrast callout: *large conditional mechanism* vs *small incremental ranking info* (why B follows from A).
- D ✅ operating points as risk/trust zones: n_ind=0 → P=.094 (reject, 34% coverage); n_ind≥2 → .77 (trust, 25%). `e26 operating_points`
- E ✅ selector **ΔAcc heatmap** config×benchmark (positive-sig / neutral-grey / negative; neutral≈white). `e25_{generalization,newbench,scale}…json`
- F ✅ gain vs **headroom**: x=CR_history−maj, y=ΔAcc_selector. `e25 + e18`
- G ✅ gain vs **identifiability**: x=provenance AUC (combo/n_indep), y=ΔAcc. `e25`
- H ✅ **applicability landscape**: x=identifiability, y=availability(CR_history), cells placed, top-right=high-leverage; principle `useful ⟸ identifiability ∧ availability`. `e25`
  *(Selector "ladder" of the 5 voting rules stays the master TABLE `tab:hpmad`, NOT re-plotted here or in Fig 7.)*

### Fig 6 — Generalization atlas (evidence atlas, not one big heatmap) · 7 panels
Thesis: *the two scientific claims travel; utility is regime-bound.* Owns ALL cross-cell grids + scale + N.
- A ✎ coverage map: 5 families (Qwen/Phi/Llama/Mistral/Gemma) on a 3B──14B axis, N=3──5, 8 benches.
- B ✅ **H1 matrix** (config×bench, extinction sign). `e25…`
- C ✅ **H2 matrix** (same grid, provenance→correctness). `e25…`
- D ♻ **H3-new matrix** (same grid, independent>inherited *sign* per cell). *(per-cell β_ind recompute — ~40 fits)*
  → B/C/D aligned so a reviewer compares EXTINCTION | PROVENANCE | INDEPENDENCE at a glance.
- E ✅ scale effect 3B→7B→mixed→14B (majority↑ while remaining headroom↓ — the ceiling story). `e25 scale`
- F ✅ N=3 vs N=5 paired dots (H1/H2/H3 hold; selector lifts under N=5). `e25 scale`
- G ✅ summary bars with **correct denominators**: H1 35/40 (40/40 dir), H2 39/40, H3-new 7/8·5/5, selector 15/40.

### Fig 7 — From Influence to Provenance (journal-vs-conference bridge) · 5 panels · [foldable → Intro]
Thesis: *don't ask who supplied an answer; ask how many independent paths produced it.*
- A ✎ conference agent-network "who influences whom".
- B ✅ influence-only mechanism AUC ≈ 0.49 (chance) vs structural control 0.63. `e2…json`
- C ✎ two hypotheses, same influential agent, different provenance → influence can't tell them apart.
- D ✎ hypothesis-centric provenance graph (the journal view).
- E ✎ conceptual shift `agent importance → evidence provenance`.
  *(influence-vote < majority number lives in the master table, not re-plotted.)*

---

## SUPPLEMENTARY (dedup: main = summary, supp = full detail)
- **S1 Extinction anatomy** (5): social-adoption vs self-error, |A| confound, 73% surface-adoption. `e4a…`
- **S2 Predictability ≠ actionability** (6): hazard ROC/AUPRC, LOBO; oracle +16.5pp vs learned CATE≈random (the 5 negatives). `e3,e13,e11…` — main-text home is the Fig-7 bridge + prose.
- **S3 Intervention / rescue** (5): E5 counterfactual preserve +12/+16pp. `e5…`
- **S4 E27 genuine vs surface** (4): naive .356>.292 → controlled β_gen .481 ≈ β_surf .495, diff −.014[−.111,.080], independent 1.546 towers. `e27…` *(already have `e27_surface_genuine.png`)*
- **S5 Full per-benchmark E26** (8): all features × all benches (Fig 4F is only the β_ind headline). `e26 recompute`
- **S6 Full per-config/model E26** (5): Fig 4G expanded. `e26 recompute`
- **S7 Selector baselines + ablations** (7): round0/freq/persistence/recency + E23 ablation + E24 convergence + E21 judge-fails. `e19,e23,e24,e21…`
- **S8 Statistical robustness** (5): cluster-bootstrap dists, per-qid sample sizes, tie-break sensitivity, majority-baseline unification (0.509 vs 0.528). `e18/e23…`

---

## De-duplication ledger (what moved, so nothing repeats)
- Extinction bench×config heatmap: **removed from Fig 2** → lives in **Fig 6B (H1 matrix)**.
- Scale co-shrink: **removed from Fig 2** → **Fig 6E**.
- H2 generalization mini-matrix: **removed from Fig 3** → **Fig 6C**.
- Collinearity + incremental-AUC + operating points: **not in Fig 4** (hero stays pure mechanism) → **Fig 5A/B/D**.
- Old Fig 5 (why-modest) + old Fig 6 (when-useful) → **merged into one Fig 5 (decision utility)**.
- Selector 5-rule ladder (incl influence-vote): **table `tab:hpmad` only** — not a panel in Fig 5 or Fig 7.
- Fig 4F/G (β magnitudes + CI, marginals) vs Fig 6D (per-cell *sign*): different granularity/abstraction — kept both deliberately, not a repeat.
- Fig 4B (heatmap) vs 4C (curves): two reads of the ONE signature result, hero-only — intentional, not a repeat.

## Data-readiness summary (the "đúng hết chưa" check)
- **Drawable now (✅):** Fig 2B-D, 3B/E/F, 4B/C/E/F/G, 5B/D/E/F/G/H, 6B/C/E/F/G, 7B, S1-S4, S7-S8. All backed by frozen JSON already on disk.
- **Cheap recompute (♻, zero-GPU, from the E26 candidate table):** Fig 3A fingerprint means, 3C/D ribbons, 4D two intermediate control rungs, 5A scatter, 6D per-cell H3-new β, S5/S6. → one small analysis pass produces all ♻ panels.
- **Conceptual/design (✎):** Fig 1A-C, 2A trajectory (pick GSM8K 42→48 log), 4A paradox, 4H strip, 6A coverage map, 7A/C/D/E. Need illustration + one representative log extract, no stats risk.
- **Gap flagged:** RQ2/failure≠benefit has NO main-text figure (demoted to bridge) — represented by Fig 7 + S2. If a reviewer wants a main-text nod, add one oracle-vs-CATE panel to Fig 7. Your call.

## Priority (invest polish in this order)
**Fig 4 (money) ≫ Fig 2 ≈ Fig 1 > Fig 3 > Fig 5 > Fig 6 > Fig 7.** If only 3 get master-level polish:
**Fig 1 + Fig 2 + Fig 4** (Fig 4 must be the single most beautiful figure and the graphical-abstract source).
