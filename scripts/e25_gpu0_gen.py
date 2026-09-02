"""E25 model-scale + N-scaling PILOT generation on gpu0 (:11436).

Prereg ADDENDUM: paper/E25_newbench_prereg.md (locked 2026-08-31). Method FROZEN at E23 — this only
generates debates for two NEW ensemble conditions across the full 8-benchmark set, then the frozen
e18/e22/e23 verdict (via experiments/e25_newbench_verdict.py) is reused unchanged.

  E_strong = {Qwen2.5-14B, Phi4-mini, Llama3.2-3B}         (strong-lite anchor, 13.5 GB, fits 16 GB)
  E_n5     = {Qwen3B, Phi4-mini, Llama3B, Mistral7B, Qwen7B} (N=5 stress test, 15.5 GB)

GPU discipline ([[gpu-run-pinning]]): gpu0-pinned Ollama :11436 ONLY (CUDA_VISIBLE_DEVICES=0,
OLLAMA_VULKAN=0). Leave the gpu1 :11435 new-bench pilot PID untouched. Resumable (gen_cell skips
done question_ids). Ordered by ensemble to keep each ensemble's models hot across its 8 benches.

  /home/alex/venvs/env/bin/python scripts/e25_gpu0_gen.py
"""
import os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import experiments.e25_generalization as e25

GPU0_URL = "http://localhost:11436"
SEED, N = 42, 100
BENCHES = ["gsm8k", "strategyqa", "commonsenseqa", "bbh", "truthfulqa", "aqua", "arc", "mmlu"]
# ensemble-major order keeps E_strong's 3 models resident for its 8 benches, then E_n5's 5.
CELLS = [("E_strong", b) for b in BENCHES] + [("E_n5", b) for b in BENCHES]


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


if __name__ == "__main__":
    log(f"=== E25 gpu0 gen (E_strong + E_n5): {len(CELLS)} cells, seed{SEED} n={N}, url={GPU0_URL} ===")
    for i, (ens, bench) in enumerate(CELLS, 1):
        log(f">>> ({i}/{len(CELLS)}) {ens}/{bench}")
        try:
            e25.gen_cell(ens, bench, SEED, N, backend="ollama", base_url=GPU0_URL)
            log(f"<<< DONE {ens}/{bench}")
        except Exception as e:
            log(f"!!! FAIL {ens}/{bench}: {e}")
    log("=== ALL gpu0 PILOT GEN DONE ===")
