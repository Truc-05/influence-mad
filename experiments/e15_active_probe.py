"""
E15 — Active Diagnostic Probe generation (GPU, gpu1-pinned :11435).
See paper/E15_active_probe_prereg.md (LOCKED). Generates, for each (event, agent) in the
E13/E14 population, ONE cheap self-diagnosis probe (no answer revision). Raw responses are
cached per bench:qid:seed:agent so the run is resumable. Evaluation is separate
(e15_eval.py, CPU) and reuses the existing per-seed ITE labels.

Usage:
  python experiments/e15_active_probe.py --smoke 2      # 2 events x 3 agents, gpu1
  python experiments/e15_active_probe.py                # full generation (resumable)
"""
import os, sys, argparse, pickle
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.io_utils import load_config
import experiments.e5_counterfactual as e5
import experiments.e6_offline as off

CACHE = "results/cache/e15_probe.pkl"
CONFIG = "config/config_e6_gpu1.yaml"     # base_url :11435 (gpu1); NEVER :11434

PROBE = """{persona}

Question: {question}

Your current answer: {answer}
Your reasoning: {reasoning}

Before any revision, critically examine your OWN answer and name the single most likely
specific flaw that could make it incorrect. Do NOT change your answer.
Respond exactly in this format:
FLAW: <one specific flaw, or "none" if you are confident there is no error>
CONFIDENCE: <a number 0-100 for how confident you are the answer is correct>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    config = load_config(args.config)
    agents = e5.build_agents(config)
    events = off.load_events()
    if args.smoke:
        events = events[:args.smoke]
    cache = pickle.load(open(args.cache, "rb")) if os.path.exists(args.cache) else {}
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    todo = [(ev, n) for ev in events for n in ev["names"]
            if cache.get(f"{ev['bench']}:{ev['qid']}:{ev['seed']}:{n}") in (None, "__ERROR__")]
    print(f"population {len(events)} events; probe calls to do: {len(todo)} "
          f"(cached {len(cache)})")
    done = 0
    for ev, name in todo:
        key = f"{ev['bench']}:{ev['qid']}:{ev['seed']}:{name}"
        ag = agents[name]
        prompt = PROBE.format(persona=ag.persona_text, question=ev["question"],
                              answer=ev["trigger_state"][name]["answer"],
                              reasoning=ev["trigger_state"][name]["reasoning"])
        try:
            cache[key] = ag._call_model(prompt)
        except Exception as e:
            cache[key] = "__ERROR__"; print(f"  ERROR {key}: {e}")
        done += 1
        if done % 20 == 0:
            pickle.dump(cache, open(args.cache, "wb"))
            print(f"  probe [{done}/{len(todo)}]")
    pickle.dump(cache, open(args.cache, "wb"))
    ok = sum(1 for v in cache.values() if v not in (None, "__ERROR__"))
    print(f"[OK] {args.cache}  ({ok} good, {sum(1 for v in cache.values() if v=='__ERROR__')} errors)")
    if args.smoke:
        for k, v in list(cache.items())[:3]:
            print(f"\n--- {k} ---\n{v[:300]}")


if __name__ == "__main__":
    main()
