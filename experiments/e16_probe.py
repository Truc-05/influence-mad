"""
E16 — Placebo-controlled dose-response probe GENERATION (GPU, gpu1-pinned :11435).
See paper/E16_dose_response_prereg.md (LOCKED). For each (event, agent) in the frozen
E13/E15 population, generate ONE re-query round under two matched arms:
    P0 = placebo           (e5.CLAUSE["control"] = "")
    P1 = low-dose diagnostic (e5.CLAUSE["combined"] = self_verify + influence_resistant)
Both are round-1 (t_star=1) from the logged gate-0 trigger_state, using the identical
e5._REFINE prompt + peer block — matched on calls / length / reflection, differing ONLY by
the mechanism clause. D1 is a genuine dose-reduction of the stored `combined` treatment
(round-1 vs run-to-consensus). Outcomes are NOT regenerated (reuse per-seed ITE labels).

Resumable cache keyed bench:qid:seed -> {"P0":{name:{answer,reasoning}}, "P1":{...}}.
Evaluation is separate & CPU (e16_eval.py). GENERATION IS gpu1-ONLY; NEVER :11434 / gpu0.

Usage:
  python experiments/e16_probe.py --selftest      # stub gen, NO GPU (plumbing check)
  python experiments/e16_probe.py --smoke 2       # 2 events x 2 arms x 3 agents, gpu1
  python experiments/e16_probe.py                 # full generation (resumable)
"""
import os, sys, argparse, pickle
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.io_utils import load_config
from src.agents.base_agent import AgentResponse
import experiments.e5_counterfactual as e5
import experiments.e6_offline as off

CACHE = "results/cache/e16_probe.pkl"
CONFIG = "config/config_e6_gpu1.yaml"          # base_url :11435 (gpu1); NEVER :11434
ARMS = {"P0": e5.CLAUSE["control"], "P1": e5.CLAUSE["combined"]}   # placebo / low-dose diagnostic


def probe_event(ev, agents, gen_fn):
    """One re-query round per arm from the gate trigger_state. Returns
    {"P0":{name:{answer,reasoning}}, "P1":{...}}. Exactly the first iteration of
    e5.run_arm for each clause (no forward continuation)."""
    names, q = ev["names"], ev["question"]
    prev = {n: dict(ev["trigger_state"][n]) for n in names}
    out = {}
    for arm, clause in ARMS.items():
        curr = {}
        for n in names:
            peers = {o: prev[o] for o in names if o != n}
            prompt = e5.build_prompt(agents[n].persona_text, q, prev[n], peers, clause)
            resp = gen_fn(agents[n], prompt)
            curr[n] = {"answer": resp.answer, "reasoning": resp.reasoning}
        out[arm] = curr
    return out


def _key(ev):
    return f"{ev['bench']}:{ev['qid']}:{ev['seed']}"


def _valid(v):
    return isinstance(v, dict) and set(v) == set(ARMS) and all(
        isinstance(v[a], dict) and v[a] for a in ARMS)


def run(events, agents, gen_fn, cache_path, save_every=10):
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    todo = [ev for ev in events if not _valid(cache.get(_key(ev)))]
    print(f"population {len(events)} events; dose-probe events to do: {len(todo)} "
          f"(cached {sum(1 for v in cache.values() if _valid(v))})")
    for i, ev in enumerate(todo):
        try:
            cache[_key(ev)] = probe_event(ev, agents, gen_fn)
        except Exception as e:
            cache[_key(ev)] = "__ERROR__"; print(f"  ERROR {_key(ev)}: {e}")
        if (i + 1) % save_every == 0 or i + 1 == len(todo):
            pickle.dump(cache, open(cache_path, "wb"))
            print(f"  dose-probe [{i+1}/{len(todo)}]")
    ok = sum(1 for v in cache.values() if _valid(v))
    err = sum(1 for v in cache.values() if v == "__ERROR__")
    print(f"[OK] {cache_path}  ({ok} good, {err} errors)")
    return cache


# ------------------------------- selftest (NO GPU) ----------------------------
def selftest():
    """Stub generator: verifies (a) both arms produced for every agent, (b) placebo vs
    diagnostic are recorded separately, (c) cache round-trips and is resumable. No GPU."""
    events = off.load_events()[:3]
    names = events[0]["names"]
    print(f"[selftest] {len(events)} events, agents={names}")

    class Stub:
        def __init__(self, name): self.persona_text = f"stub-{name}"
    agents = {n: Stub(n) for n in names}

    calls = {"c": 0}
    def gen(agent, prompt):
        calls["c"] += 1
        # deterministic stub: diagnostic clause -> pretend the agent moves; placebo -> keeps.
        moved = "Re-derive the answer independently" in prompt
        return AgentResponse(answer=("moved" if moved else "kept"),
                             reasoning=("changed reasoning" if moved else "same reasoning"),
                             raw_text="")

    tmp = "results/cache/_e16_selftest.pkl"
    if os.path.exists(tmp):
        os.remove(tmp)
    cache = run(events, agents, gen, tmp, save_every=1)
    assert calls["c"] == len(events) * 2 * len(names), "calls != events*arms*agents"
    for ev in events:
        v = cache[_key(ev)]
        assert _valid(v), "arm structure invalid"
        assert set(v["P0"]) == set(names) and set(v["P1"]) == set(names)
        assert all(v["P1"][n]["answer"] == "moved" for n in names), "diagnostic arm not distinct"
        assert all(v["P0"][n]["answer"] == "kept" for n in names), "placebo arm not distinct"
    # resumability: a second run does zero work
    before = calls["c"]
    run(events, agents, gen, tmp, save_every=1)
    assert calls["c"] == before, "resume regenerated cached events"
    os.remove(tmp)
    print(f"[selftest] calls={before} (= {len(events)}*2*{len(names)}); arms distinct; "
          f"resume=no-op. PLUMBING VALID (NO GPU).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    if args.selftest:
        selftest(); return

    config = load_config(args.config)
    assert "11435" in config["debate"]["base_url"], \
        f"E16 must run on gpu1 :11435, got {config['debate']['base_url']}"
    agents = e5.build_agents(config)
    events = off.load_events()
    if args.smoke:
        events = events[:args.smoke]
    gen_fn = lambda agent, prompt: agent.generate(prompt)
    cache = run(events, agents, gen_fn, args.cache)
    if args.smoke:
        for k, v in list(cache.items())[:2]:
            if _valid(v):
                n0 = list(v["P0"])[0]
                print(f"\n--- {k} [{n0}] ---")
                print(f"  P0: {v['P0'][n0]}")
                print(f"  P1: {v['P1'][n0]}")


if __name__ == "__main__":
    main()
