"""E25 FRONTIER (cloud-API) ensemble generation. ZERO local GPU -> no conflict with E_strong24.

E_frontier = 3 personas (careful/skeptic/bigpicture) all backed by a frontier cloud model over an
OpenAI-compatible API (DeepSeek by default). Homogeneous backbone keeps it PURE API (no Ollama, no GPU);
round-0 diversity comes from the personas + temperature. Method FROZEN at E23; the frozen e18/e22/e23
verdict (via experiments/e25_scale_verdict.py) is reused unchanged. Directly answers the "no frontier
model" reviewer objection (report as a frontier-scale API model, not "up to 14B/24B open-weight").

Requires (set before launch):
  export DEEPSEEK_API_KEY=sk-...            # cloud API key (never committed)
  export DEEPSEEK_MODEL=deepseek-chat       # EXACT model id (confirm the "v4 flash" id first)
  export DEEPSEEK_BASE_URL=https://api.deepseek.com   # optional, this is the default

Cost: paid API. Stage 1 (8 benches x 100 q, ~8.7 calls/debate) ~= 7k calls. Resumable (skips done qids).

  DEEPSEEK_API_KEY=... DEEPSEEK_MODEL=... python scripts/e25_frontier_gen.py [--smoke]
"""
import os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import experiments.e25_generalization as e25

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
BACKEND = "deepseek"
ENS = "E_frontier"
N = 100
CORE = ["gsm8k", "commonsenseqa", "bbh", "strategyqa", "truthfulqa"]
BREADTH = ["aqua", "arc", "mmlu"]
CONFIRM = ["gsm8k", "commonsenseqa", "bbh", "strategyqa"]
STAGE1 = [(b, 42) for b in CORE + BREADTH]
STAGE2 = [(b, s) for s in (43, 44) for b in CONFIRM]
CELLS = STAGE1 + STAGE2


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("ERROR: set $DEEPSEEK_API_KEY first")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    if "--smoke" in sys.argv:
        log(f"SMOKE: 2 debates, {ENS} model={model} url={BASE_URL}")
        e25.gen_cell(ENS, "gsm8k", 42, 2, backend=BACKEND, base_url=BASE_URL)
        log("SMOKE DONE — inspect data/debate_logs/gen/E_frontier/gsm8k_seed42.jsonl")
        sys.exit(0)
    log(f"=== E25 {ENS} gen: {len(CELLS)} cells, model={model}, url={BASE_URL} ===")
    for i, (bench, seed) in enumerate(CELLS, 1):
        log(f">>> ({i}/{len(CELLS)}) {ENS}/{bench}_seed{seed}")
        try:
            e25.gen_cell(ENS, bench, seed, N, backend=BACKEND, base_url=BASE_URL)
            log(f"<<< DONE {ENS}/{bench}_seed{seed}")
        except Exception as e:
            log(f"!!! FAIL {ENS}/{bench}_seed{seed}: {e}")
    log(f"=== ALL {ENS} GEN DONE ===")
