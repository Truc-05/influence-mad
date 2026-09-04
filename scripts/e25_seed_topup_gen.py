"""Seed top-up for E_strong (14B) + E_n5 (N=5) — closes reviewer objection #2 ("those two cells,
used to argue the effect survives 14B/N=5, ran only 1 seed"). Adds seeds 43-44 on the confirmatory
core {gsm8k, commonsenseqa, bbh, strategyqa} so each of those cells gets 3 seeds.

GPU discipline: shares gpu0 :11436 with E_strong24, so it WAITS for e25_strong24_gen.py to exit before
starting (no contention). Detached, resumable (gen_cell skips done qids). Method FROZEN at E23.

  CUDA_VISIBLE_DEVICES="" setsid nohup python scripts/e25_seed_topup_gen.py \
      > results/logs/e25_seed_topup.log 2>&1 &
"""
import os, sys, time, subprocess
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import experiments.e25_generalization as e25

GPU0_URL = "http://localhost:11436"
CORE = ["gsm8k", "commonsenseqa", "bbh", "strategyqa"]
ENSEMBLES = ["E_strong", "E_n5"]
SEEDS = [43, 44]
N = 100


def running(name):
    return subprocess.run(["pgrep", "-f", name], capture_output=True).returncode == 0


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


if __name__ == "__main__":
    log("seed top-up queued; waiting for E_strong24 to finish (shares gpu0 :11436)...")
    while running("e25_strong24_gen.py"):
        time.sleep(300)
    log("E_strong24 finished — starting seed top-up (E_strong 14B + E_n5, seeds 43-44, core benches)")
    cells = [(e, b, s) for e in ENSEMBLES for s in SEEDS for b in CORE]  # 2 x 4 x 2 = 16 cells
    for i, (ens, bench, seed) in enumerate(cells, 1):
        log(f">>> ({i}/{len(cells)}) {ens}/{bench}_seed{seed}")
        try:
            e25.gen_cell(ens, bench, seed, N, backend="ollama", base_url=GPU0_URL)
            log(f"<<< DONE {ens}/{bench}_seed{seed}")
        except Exception as e:
            log(f"!!! FAIL {ens}/{bench}_seed{seed}: {e}")
    log("=== ALL seed top-up GEN DONE (E_strong + E_n5 now 3 seeds on core) ===")
