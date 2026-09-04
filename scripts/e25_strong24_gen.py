"""E25 LARGER-ANCHOR (24B) ensemble generation on gpu0 (:11436).

E_strong24 = {mistral-small3.1:24b, phi4-mini, llama3.2:3b} parallels E_strong (14B -> 24B anchor)
to push the tested range from 3B--14B up to 3B--24B, blunting the reviewer's "only tiny models"
objection. Method is FROZEN at E23 -- this only generates debates; the frozen e18/e22/e23 verdicts
(via experiments/e25_scale_verdict.py) are reused unchanged.

GPU discipline ([[gpu-run-pinning]]): gpu0-pinned Ollama :11436 ONLY (CUDA_VISIBLE_DEVICES=0). Leave
the gpu1 job (:11445 / :11435) untouched. The 24B anchor is 15 GB and cannot co-reside with the two
small peers in 16 GB, so Ollama swaps it per round (~75 s/debate measured). Resumable: gen_cell skips
done question_ids, so a restart continues where it stopped.

Stage 1 = all 8 benches at seed 42 (matches E_strong/E_n5 breadth). Core benches first so the most
important cells land early. Stage 2 = seeds 43-44 on the 4 confirmatory core benches (multi-seed
robustness for the new anchor -- directly answers the "E_strong/E_n5 only 1 seed" objection).

  CUDA_VISIBLE_DEVICES="" setsid nohup python scripts/e25_strong24_gen.py \
      > results/logs/e25_strong24.log 2>&1 &
"""
import os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import experiments.e25_generalization as e25

GPU0_URL = "http://localhost:11436"
ENS = "E_strong24"
N = 100
CORE = ["gsm8k", "commonsenseqa", "bbh", "strategyqa", "truthfulqa"]
BREADTH = ["aqua", "arc", "mmlu"]
CONFIRM = ["gsm8k", "commonsenseqa", "bbh", "strategyqa"]
STAGE1 = [(b, 42) for b in CORE + BREADTH]                     # 8 cells
STAGE2 = [(b, s) for s in (43, 44) for b in CONFIRM]           # 8 cells (multi-seed core)
CELLS = STAGE1 + STAGE2


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


if __name__ == "__main__":
    log(f"=== E25 {ENS} gen: {len(CELLS)} cells (stage1={len(STAGE1)}, stage2={len(STAGE2)}), "
        f"n={N}, url={GPU0_URL} ===")
    for i, (bench, seed) in enumerate(CELLS, 1):
        log(f">>> ({i}/{len(CELLS)}) {ENS}/{bench}_seed{seed}")
        try:
            e25.gen_cell(ENS, bench, seed, N, backend="ollama", base_url=GPU0_URL)
            log(f"<<< DONE {ENS}/{bench}_seed{seed}")
        except Exception as e:
            log(f"!!! FAIL {ENS}/{bench}_seed{seed}: {e}")
    log(f"=== ALL {ENS} GEN DONE ===")
