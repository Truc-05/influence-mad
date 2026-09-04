import os
import json
import random

from src.agents.peer_agent import PeerAgent
from src.data_collection.log_schema import DebateLog, RoundLog
from src.utils.io_utils import read_jsonl


def _norm(text):
    return (text or "").strip().lower()


def _build_agents(config):
    """Dựng dict {name: PeerAgent} từ config['agents']."""
    debate_cfg = config["debate"]
    agents = {}
    for spec in config["agents"]:
        agents[spec["name"]] = PeerAgent(
            model_name=spec["model"],
            role=spec["name"],
            temperature=spec.get("temperature", debate_cfg.get("temperature", 0.3)),
            base_url=debate_cfg["base_url"],
            max_tokens=debate_cfg.get("max_tokens", 512),
            timeout=debate_cfg.get("timeout_seconds", 180),
            persona=spec.get("persona", "careful"),
            backend=debate_cfg.get("backend", "ollama"),
            api_key_env=debate_cfg.get("api_key_env", "DEEPSEEK_API_KEY"),
        )
    return agents


def _consensus(answers):
    return len({_norm(a) for a in answers.values()}) == 1


def _run_single_debate(question_id, question, ground_truth, agents, max_rounds,
                       correct_answers=None):
    """Debate đồng bộ giữa các peer: mỗi vòng tất cả agent trả lời dựa trên
    đáp án của vòng TRƯỚC (synchronous update)."""
    log = DebateLog(question_id=question_id, question=question, ground_truth=ground_truth,
                    correct_answers=list(correct_answers or []))
    names = list(agents.keys())

    prev_answers = {}
    prev_reasonings = {}

    for round_idx in range(max_rounds):
        curr_answers = {}
        curr_reasonings = {}

        for name in names:
            agent = agents[name]
            if round_idx == 0:
                resp = agent.solve(question)
            else:
                peers = {
                    other: {"answer": prev_answers[other], "reasoning": prev_reasonings[other]}
                    for other in names if other != name
                }
                resp = agent.refine(question, prev_answers[name], prev_reasonings[name], peers)
            curr_answers[name] = resp.answer
            curr_reasonings[name] = resp.reasoning

        changed = {
            name: (round_idx > 0 and _norm(curr_answers[name]) != _norm(prev_answers[name]))
            for name in names
        }

        log.rounds.append(
            RoundLog(
                round_idx=round_idx,
                answers={n: {"answer": curr_answers[n], "reasoning": curr_reasonings[n]} for n in names},
                changed=changed,
            )
        )

        if round_idx == 0:
            log.independent_answers = dict(curr_answers)

        prev_answers = curr_answers
        prev_reasonings = curr_reasonings

        if round_idx > 0 and _consensus(prev_answers):
            break

    log.final_answers = dict(prev_answers)
    return log


_MOCK_SKILL = {"careful": 0.75, "skeptic": 0.60, "bigpicture": 0.50}
_MOCK_SUSCEPT = {"careful": 0.15, "skeptic": 0.35, "bigpicture": 0.55}


def _mock_single_debate(question_id, question, ground_truth, agent_specs, max_rounds, rng,
                        correct_answers=None):
    names = [s["name"] for s in agent_specs]
    persona = {s["name"]: s.get("persona", "careful") for s in agent_specs}
    correct = str(ground_truth)

    log = DebateLog(question_id=question_id, question=question, ground_truth=ground_truth,
                    correct_answers=list(correct_answers or []))

    prev = {}
    for round_idx in range(max_rounds):
        curr = {}
        for name in names:
            p = persona[name]
            if round_idx == 0:
                skill = _MOCK_SKILL.get(p, 0.55)
                curr[name] = correct if rng.random() < skill else f"wrong_{rng.randint(1, 3)}"
            else:
                others = [prev[o] for o in names if o != name]

                maj = max(set(others), key=others.count)
                susc = _MOCK_SUSCEPT.get(p, 0.35)
                if _norm(prev[name]) != _norm(maj) and rng.random() < susc:
                    curr[name] = maj                 
                elif rng.random() < 0.10:
                    curr[name] = correct                  
                else:
                    curr[name] = prev[name]

        changed = {n: (round_idx > 0 and _norm(curr[n]) != _norm(prev[n])) for n in names}
        log.rounds.append(
            RoundLog(
                round_idx=round_idx,
                answers={n: {"answer": curr[n], "reasoning": "mock"} for n in names},
                changed=changed,
            )
        )
        if round_idx == 0:
            log.independent_answers = dict(curr)
        prev = curr
        if round_idx > 0 and _consensus(prev):
            break

    log.final_answers = dict(prev)
    return log


def _load_existing(output_path):
    if not os.path.exists(output_path):
        return [], set()
    logs = read_jsonl(output_path)
    done_ids = {log["question_id"] for log in logs}
    return logs, done_ids


def run_debate_on_benchmark(benchmark_name, n_samples, config, resume=True, seed=None):
    """Thu thập debate log. Nếu truyền seed, log ghi vào file riêng theo seed
    (phục vụ multi-seed). backend='mock' -> giả lập offline."""
    processed_path = f"data/processed/{benchmark_name}.jsonl"
    suffix = "" if seed is None else f"_seed{seed}"
    output_path = f"data/debate_logs/{benchmark_name}{suffix}.jsonl"

    data = read_jsonl(processed_path)[:n_samples]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if resume:
        logs, done_ids = _load_existing(output_path)
    else:
        logs, done_ids = [], set()

    remaining = [item for item in data if item["question_id"] not in done_ids]
    if not remaining:
        print(f"{benchmark_name}{suffix}: already complete ({len(logs)}/{len(data)}), nothing to run")
        return logs

    print(f"{benchmark_name}{suffix}: {len(done_ids)} done, {len(remaining)} remaining out of {len(data)}")

    backend = config["debate"].get("backend", "ollama")
    max_rounds = config["debate"]["max_rounds"]

    agents = None
    if backend != "mock":
        agents = _build_agents(config)

    with open(output_path, "a", encoding="utf-8") as f:
        for idx, item in enumerate(remaining):
            try:
                if backend == "mock":
                    rng = random.Random(f"{seed}-{item['question_id']}")
                    log = _mock_single_debate(
                        item["question_id"], item["question"], item["ground_truth"],
                        config["agents"], max_rounds, rng,
                        correct_answers=item.get("correct_answers"),
                    )
                else:
                    log = _run_single_debate(
                        item["question_id"], item["question"], item["ground_truth"],
                        agents, max_rounds,
                        correct_answers=item.get("correct_answers"),
                    )
            except Exception as e:
                print(f"[{idx + 1}/{len(remaining)}] {item['question_id']} FAILED: {e}")
                continue

            log_dict = log.to_dict()
            logs.append(log_dict)
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
            f.flush()

    print(f"{benchmark_name}{suffix}: collected {len(logs)} logs")
    return logs
