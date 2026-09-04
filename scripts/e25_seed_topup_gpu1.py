"""Seed top-up for E_strong (14B) + E_n5 (N=5), seeds 43-44 on core {gsm8k, commonsenseqa, bbh,
strategyqa} — closes reviewer objection #2. IDENTICAL cells to e25_seed_topup_gen.py, but pinned to
GPU 1 (ollama :11435, CUDA_VISIBLE_DEVICES=1) and runs IMMEDIATELY (no wait) so it goes in PARALLEL
with E_strong24 on GPU 0. Method FROZEN at E23. Resumable (gen_cell skips done qids), detached.

  CUDA_VISIBLE_DEVICES="" setsid nohup python scripts/e25_seed_topup_gpu1.py \
      > results/logs/e25_seed_topup_gpu1.log 2>&1 &
"""
import os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import experiments.e25_generalization as e25

GPU1_URL = "http://localhost:11435"   # ollama server pinned to GPU 1 (CUDA_VISIBLE_DEVICES=1)
CORE = ["gsm8k", "commonsenseqa", "bbh", "strategyqa"]
ENSEMBLES = ["E_strong", "E_n5"]
SEEDS = [43, 44]
N = 100


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


if __name__ == "__main__":
    log(f"seed top-up on GPU 1 ({GPU1_URL}) — running in PARALLEL with E_strong24 (GPU 0)")
    cells = [(e, b, s) for e in ENSEMBLES for s in SEEDS for b in CORE]  # 2 x 4 x 2 = 16 cells
    for i, (ens, bench, seed) in enumerate(cells, 1):
        log(f">>> ({i}/{len(cells)}) {ens}/{bench}_seed{seed}")
        try:
            e25.gen_cell(ens, bench, seed, N, backend="ollama", base_url=GPU1_URL)
            log(f"<<< DONE {ens}/{bench}_seed{seed}")
        except Exception as e:
            log(f"!!! FAIL {ens}/{bench}_seed{seed}: {e}")
    log("=== ALL seed top-up (GPU1) GEN DONE (E_strong + E_n5 now 3 seeds on core) ===")
