"""
Chạy pipeline qua NHIỀU seed để đo độ ổn định (mean ± std) của Influence Score,
Accuracy per agent, và Voting accuracy.

Với mỗi seed: thu thập debate log riêng -> build Influence Matrix -> PageRank ->
correlation/classify -> đánh giá 3 chiến lược voting. Sau đó tổng hợp mean/std.

Usage:
    python experiments/run_multiseed.py --benchmark gsm8k
    python experiments/run_multiseed.py --benchmark gsm8k --seeds 42 43 44 45 46
    python experiments/run_multiseed.py --benchmark gsm8k --backend mock
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io_utils import load_config, read_jsonl
from src.utils.seed import set_seed
from src.utils.scoring import is_correct

from src.data_collection.run_debate import run_debate_on_benchmark
from src.influence_estimation.influence_matrix import build_influence_matrix, get_agent_names
from src.influence_estimation.pagerank_score import compute_influence_scores
from src.influence_analysis.correlation import compute_influence_accuracy_correlation, accuracy_per_agent
from src.influence_analysis.classify_influence import classify_agents
from src.influence_voting.strategies import majority_voting, influence_based_voting, influence_aware_voting


def _get_n_samples(config, benchmark_name):
    if config.get("_n_samples_override"):
        return config["_n_samples_override"]
    for b in config["benchmarks"]:
        if b["name"] == benchmark_name:
            return b["n_samples"]
    raise ValueError(f"benchmark {benchmark_name} not found in config")


def _voting_accuracy(debate_logs, influence_scores, classification, benchmark_name, agent_names):
    res = {"majority": 0, "influence_based": 0, "influence_aware": 0}
    total = len(debate_logs)
    for log in debate_logs:
        finals = log.get("final_answers", {})
        cand = {n: finals[n] for n in agent_names if n in finals}
        if not cand:
            continue
        gt = log["ground_truth"]
        ca = log.get("correct_answers")
        if is_correct(majority_voting(cand), gt, benchmark_name, ca):
            res["majority"] += 1
        if is_correct(influence_based_voting(cand, influence_scores), gt, benchmark_name, ca):
            res["influence_based"] += 1
        if is_correct(influence_aware_voting(cand, influence_scores, classification), gt, benchmark_name, ca):
            res["influence_aware"] += 1
    return {k: (v / total if total else 0.0) for k, v in res.items()}


def _load_correct_answers_map(benchmark_name):
    """Map question_id -> correct_answers từ data/processed (nếu có).

    Cho phép chấm lại các debate log CŨ (được ghi trước khi log mang theo
    correct_answers) mà không cần chạy lại debate."""
    path = f"data/processed/{benchmark_name}.jsonl"
    if not os.path.exists(path):
        return {}
    out = {}
    for rec in read_jsonl(path):
        ca = rec.get("correct_answers")
        if ca:
            out[rec["question_id"]] = ca
    return out


def _run_one_seed(benchmark_name, seed, config):
    set_seed(seed)
    n_samples = _get_n_samples(config, benchmark_name)

    # Luôn gọi run_debate_on_benchmark: nó tự resume theo question_id (bỏ qua
    # câu đã ghi, chỉ chạy phần còn thiếu). Nhờ vậy nếu tắt máy giữa chừng, lần
    # chạy sau tiếp tục đúng chỗ dừng thay vì tưởng seed đã xong.
    debate_logs = run_debate_on_benchmark(benchmark_name, n_samples, config, resume=True, seed=seed)

    # Bơm correct_answers vào các log còn thiếu field này (log cũ), lấy từ
    # data/processed theo question_id. Nếu processed chưa có correct_answers thì
    # bỏ qua, chấm điểm quay về chỉ dùng ground_truth.
    ca_map = _load_correct_answers_map(benchmark_name)
    if ca_map:
        for log in debate_logs:
            if not log.get("correct_answers"):
                log["correct_answers"] = ca_map.get(log["question_id"], [])

    agent_names = get_agent_names(debate_logs)
    matrix = build_influence_matrix(debate_logs, agent_names)
    scores = compute_influence_scores(matrix, agent_names, config["influence"]["damping_factor"])
    acc = accuracy_per_agent(debate_logs, benchmark_name, agent_names)
    corr = compute_influence_accuracy_correlation(debate_logs, scores, benchmark_name)
    classification, _ = classify_agents(debate_logs, scores, benchmark_name, agent_names)
    voting = _voting_accuracy(debate_logs, scores, classification, benchmark_name, agent_names)

    return {
        "seed": seed,
        "agent_names": agent_names,
        "influence_matrix": matrix,
        "influence_scores": scores,
        "accuracy_per_agent": acc,
        "correlation": corr,
        "classification": classification,
        "voting": voting,
    }


def _mean_std_by_key(list_of_dicts, keys):
    out = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
    return out


def _plot_bars(labels, means, stds, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color="#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(benchmark_name, config_path, seeds=None, backend=None, n_samples=None):
    config = load_config(config_path)
    if backend:
        config["debate"]["backend"] = backend
    if n_samples:
        config["_n_samples_override"] = n_samples
    if seeds is None:
        seeds = config.get("seeds", [config.get("random_seed", 42)])

    per_seed = [_run_one_seed(benchmark_name, s, config) for s in seeds]
    agent_names = per_seed[0]["agent_names"]

    influence_agg = _mean_std_by_key(
        [p["influence_scores"] for p in per_seed], agent_names
    )
    accuracy_agg = _mean_std_by_key(
        [p["accuracy_per_agent"] for p in per_seed], agent_names
    )
    voting_agg = _mean_std_by_key(
        [p["voting"] for p in per_seed], ["majority", "influence_based", "influence_aware"]
    )

    # Influence matrix mean/std theo từng ô
    mats = np.stack([p["influence_matrix"] for p in per_seed], axis=0)
    matrix_mean = mats.mean(axis=0)
    matrix_std = mats.std(axis=0)

    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)
    inf_fig = _plot_bars(
        agent_names,
        [influence_agg[a]["mean"] for a in agent_names],
        [influence_agg[a]["std"] for a in agent_names],
        f"Influence Score (mean±std, {len(seeds)} seeds) - {benchmark_name}",
        "Influence Score", os.path.join(figures_dir, f"{benchmark_name}_influence_meanstd.png"),
    )
    vote_labels = ["majority", "influence_based", "influence_aware"]
    vote_fig = _plot_bars(
        vote_labels,
        [voting_agg[v]["mean"] for v in vote_labels],
        [voting_agg[v]["std"] for v in vote_labels],
        f"Voting Accuracy (mean±std, {len(seeds)} seeds) - {benchmark_name}",
        "Accuracy", os.path.join(figures_dir, f"{benchmark_name}_voting_meanstd.png"),
    )

    summary = {
        "benchmark": benchmark_name,
        "backend": config["debate"].get("backend"),
        "seeds": seeds,
        "agent_names": agent_names,
        "influence_scores": influence_agg,
        "accuracy_per_agent": accuracy_agg,
        "voting_accuracy": voting_agg,
        "influence_matrix_mean": matrix_mean.tolist(),
        "influence_matrix_std": matrix_std.tolist(),
        "per_seed": [
            {
                "seed": p["seed"],
                "influence_scores": p["influence_scores"],
                "accuracy_per_agent": p["accuracy_per_agent"],
                "voting": p["voting"],
                "classification": p["classification"],
                "correlation": p["correlation"],
            }
            for p in per_seed
        ],
        "figures": {"influence": inf_fig, "voting": vote_fig},
    }

    tables_dir = "results/tables"
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, f"{benchmark_name}_multiseed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # In gọn ra console
    print(f"\n=== {benchmark_name} | {len(seeds)} seeds | backend={config['debate'].get('backend')} ===")
    print("Influence Score (mean ± std):")
    for a in agent_names:
        print(f"  {a:12s} {influence_agg[a]['mean']:.4f} ± {influence_agg[a]['std']:.4f}"
              f"   | acc {accuracy_agg[a]['mean']:.3f} ± {accuracy_agg[a]['std']:.3f}")
    print("Voting accuracy (mean ± std):")
    for v in vote_labels:
        print(f"  {v:16s} {voting_agg[v]['mean']:.4f} ± {voting_agg[v]['std']:.4f}")
    print(f"\n[OK] Summary: {out_path}")
    print(f"[OK] Figures: {inf_fig} , {vote_fig}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--backend", default=None, help="ollama | mock (ghi đè config)")
    parser.add_argument("--n_samples", type=int, default=None, help="giới hạn số câu (vd 5 để test nhanh)")
    args = parser.parse_args()
    run(args.benchmark, args.config, args.seeds, args.backend, args.n_samples)


if __name__ == "__main__":
    main()
