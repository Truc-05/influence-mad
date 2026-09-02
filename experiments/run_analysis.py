"""
Entry-point: Influence Analysis + vẽ hình cho 1 benchmark từ debate log đã có.

Usage:
    python experiments/run_analysis.py --benchmark gsm8k
    python experiments/run_analysis.py --benchmark gsm8k --seed 42
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.io_utils import load_config, read_jsonl

from src.influence_estimation.influence_matrix import build_influence_matrix, get_agent_names
from src.influence_estimation.pagerank_score import compute_influence_scores
from src.influence_analysis.correlation import compute_influence_accuracy_correlation, accuracy_per_agent
from src.influence_analysis.classify_influence import classify_agents


def _load_debate_logs(benchmark_name, seed=None):
    suffix = "" if seed is None else f"_seed{seed}"
    log_path = f"data/debate_logs/{benchmark_name}{suffix}.jsonl"
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Khong tim thay debate log tai '{log_path}'. Hay chay data collection truoc."
        )
    return read_jsonl(log_path)


def _plot_influence_heatmap(influence_matrix, agent_names, benchmark_name, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        influence_matrix, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=agent_names, yticklabels=agent_names, vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Influenced (j)")
    ax.set_ylabel("Influencer (i)")
    ax.set_title(f"Influence Matrix - {benchmark_name}")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{benchmark_name}_influence_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_influence_vs_accuracy(influence_scores, acc, benchmark_name, output_dir):
    agents = list(influence_scores.keys())
    x = [influence_scores[a] for a in agents]
    y = [acc.get(a, 0.0) for a in agents]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=80)
    for a, xi, yi in zip(agents, x, y):
        ax.annotate(a, (xi, yi), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Influence Score (PageRank)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Influence vs Accuracy - {benchmark_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{benchmark_name}_influence_vs_accuracy.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(benchmark_name, config_path, seed=None):
    config = load_config(config_path)
    debate_logs = _load_debate_logs(benchmark_name, seed)

    agent_names = get_agent_names(debate_logs)
    influence_matrix = build_influence_matrix(debate_logs, agent_names)
    influence_scores = compute_influence_scores(
        influence_matrix, agent_names, config["influence"]["damping_factor"]
    )

    correlation = compute_influence_accuracy_correlation(debate_logs, influence_scores, benchmark_name)
    classification, acc = classify_agents(debate_logs, influence_scores, benchmark_name, agent_names)

    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)
    heatmap_path = _plot_influence_heatmap(influence_matrix, agent_names, benchmark_name, figures_dir)
    scatter_path = _plot_influence_vs_accuracy(influence_scores, acc, benchmark_name, figures_dir)

    summary = {
        "benchmark": benchmark_name,
        "seed": seed,
        "n_samples": len(debate_logs),
        "agent_names": agent_names,
        "influence_matrix": influence_matrix.tolist(),
        "influence_scores": influence_scores,
        "accuracy_per_agent": acc,
        "classification": classification,
        "correlation": correlation,
        "figures": {"heatmap": heatmap_path, "scatter": scatter_path},
    }

    tables_dir = "results/tables"
    os.makedirs(tables_dir, exist_ok=True)
    output_path = os.path.join(tables_dir, f"{benchmark_name}_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[OK] Heatmap: {heatmap_path}")
    print(f"[OK] Scatter: {scatter_path}")
    print(f"[OK] Summary: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args.benchmark, args.config, args.seed)


if __name__ == "__main__":
    main()
