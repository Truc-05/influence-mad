"""E25 breadth-extension PILOT generation (seed42, n=100) for AQuA/MMLU/ARC.

Prereg: paper/E25_newbench_prereg.md. Method FROZEN (E23) — this only generates debates for the
new benchmarks under the 3 frozen ensembles, then the (separate) verdict reuses e18/e22/e23.

GPU discipline ([[gpu-run-pinning]]): gpu1-pinned Ollama :11435 ONLY, gpu0 untouched. Resumable
(gen_cell skips already-done question_ids). Ordered by ensemble to keep each ensemble's models hot.

  python scripts/e25_newbench_gen.py
"""
import os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import experiments.e25_generalization as e25

# E_orig-3B alias: gen_cell() deliberately SKIPS the literal 'E_orig' (it reuses frozen logs).
# For the new benches E_orig has no frozen logs (except aqua seed42), so generate the 3B
# ensemble under a tagged alias into data/debate_logs/gen/E_orig3b/ — frozen originals untouched.
e25.ENSEMBLES["E_orig3b"] = e25.ENSEMBLES["E_orig"]

SEED, N = 42, 100
CELLS = (
    [("E_7b", b)     for b in ("aqua", "mmlu", "arc")] +
    [("E_mixed", b)  for b in ("aqua", "mmlu", "arc")] +
    [("E_orig3b", b) for b in ("mmlu", "arc")]      # aqua E_orig reuses frozen seed42
)


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


if __name__ == "__main__":
    log(f"=== E25 new-bench pilot gen: {len(CELLS)} cells, seed{SEED} n={N}, url={e25.GPU1_URL} ===")
    for i, (ens, bench) in enumerate(CELLS, 1):
        log(f">>> ({i}/{len(CELLS)}) {ens}/{bench}")
        try:
            e25.gen_cell(ens, bench, SEED, N, backend="ollama", base_url=e25.GPU1_URL)
            log(f"<<< DONE {ens}/{bench}")
        except Exception as e:
            log(f"!!! FAIL {ens}/{bench}: {e}")
    log("=== ALL PILOT GEN DONE ===")
