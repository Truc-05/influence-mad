import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io_utils import load_config, read_jsonl
from src.utils.seed import set_seed
from src.utils.scoring import is_correct

from src.data_collection.run_debate import run_debate_on_benchmark
from src.influence_estimation.influence_matrix import build_influence_matrix, get_agent_names
from src.influence_estimation.pagerank_score import compute_influence_scores
from src.influence_analysis.correlation import compute_influence_accuracy_correlation
from src.influence_analysis.classify_influence import classify_agents
from src.influence_voting.strategies import majority_voting, influence_based_voting, influence_aware_voting


def _get_benchmark_n_samples(config, benchmark_name):
    for b in config["benchmarks"]:
        if b["name"] == benchmark_name:
            return b["n_samples"]
    raise ValueError(f"benchmark {benchmark_name} not found in config")


def _candidate_answers(log, agent_names):
    """Đáp án cuối của từng agent làm phiếu bầu."""
    finals = log.get("final_answers", {})
    return {name: finals[name] for name in agent_names if name in finals}


def _evaluate_voting_strategies(debate_logs, influence_scores, classification, benchmark_name, agent_names):
    results = {"majority": 0, "influence_based": 0, "influence_aware": 0}
    total = len(debate_logs)
    for log in debate_logs:
        cand = _candidate_answers(log, agent_names)
        if not cand:
            continue
        gt = log["ground_truth"]
        if is_correct(majority_voting(cand), gt, benchmark_name):
            results["majority"] += 1
        if is_correct(influence_based_voting(cand, influence_scores), gt, benchmark_name):
            results["influence_based"] += 1
        if is_correct(influence_aware_voting(cand, influence_scores, classification), gt, benchmark_name):
            results["influence_aware"] += 1
    for key in results:
        results[key] = results[key] / total if total > 0 else 0.0
    return results


def run(benchmark_name, config_path, n_samples_override=None):
    config = load_config(config_path)
    set_seed(config.get("random_seed", 42))

    n_samples = n_samples_override or _get_benchmark_n_samples(config, benchmark_name)

    debate_log_path = f"data/debate_logs/{benchmark_name}.jsonl"
    if os.path.exists(debate_log_path):
        debate_logs = read_jsonl(debate_log_path)
    else:
        debate_logs = run_debate_on_benchmark(benchmark_name, n_samples, config)

    agent_names = get_agent_names(debate_logs)
    influence_matrix = build_influence_matrix(debate_logs, agent_names)
    influence_scores = compute_influence_scores(
        influence_matrix, agent_names, config["influence"]["damping_factor"]
    )

    correlation = compute_influence_accuracy_correlation(debate_logs, influence_scores, benchmark_name)
    classification, accuracy_per_agent = classify_agents(debate_logs, influence_scores, benchmark_name, agent_names)
    voting_results = _evaluate_voting_strategies(
        debate_logs, influence_scores, classification, benchmark_name, agent_names
    )

    summary = {
        "benchmark": benchmark_name,
        "n_samples": len(debate_logs),
        "agent_names": agent_names,
        "influence_scores": influence_scores,
        "accuracy_per_agent": accuracy_per_agent,
        "classification": classification,
        "correlation": correlation,
        "voting_accuracy": voting_results,
    }

    os.makedirs("results/tables", exist_ok=True)
    output_path = f"results/tables/{benchmark_name}_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    run(args.benchmark, args.config, args.n_samples)


if __name__ == "__main__":
    main()
