from scipy.stats import pearsonr, spearmanr

from src.utils.scoring import is_correct


def accuracy_per_agent(debate_logs, benchmark_name, agent_names):
    """Accuracy của từng agent dựa trên đáp án CUỐI của agent đó."""
    correct = {name: 0 for name in agent_names}
    total = {name: 0 for name in agent_names}
    for log in debate_logs:
        finals = log.get("final_answers", {})
        for name in agent_names:
            if name not in finals:
                continue
            total[name] += 1
            if is_correct(finals[name], log["ground_truth"], benchmark_name,
                          log.get("correct_answers")):
                correct[name] += 1
    return {name: (correct[name] / total[name] if total[name] else 0.0) for name in agent_names}


def compute_influence_accuracy_correlation(debate_logs, influence_scores, benchmark_name):
    """Tương quan Influence Score vs Accuracy, tính THEO AGENT (mỗi agent 1 điểm).

    Với >= 3 agent khác nhau, Pearson/Spearman đã có nghĩa (khác bản 2-agent cũ
    luôn trả None). Vẫn giữ guard: cần >= 3 điểm với influence không đồng nhất.
    """
    agents = list(influence_scores.keys())
    acc = accuracy_per_agent(debate_logs, benchmark_name, agents)

    influence_values = [influence_scores[a] for a in agents]
    accuracy_values = [acc.get(a, 0.0) for a in agents]

    result = {
        "pearson_r": None,
        "pearson_p": None,
        "spearman_r": None,
        "spearman_p": None,
        "n_agents": len(agents),
        "note": "",
    }

    if len(agents) < 3 or len(set(influence_values)) < 2:
        result["note"] = (
            f"Khong du diem du lieu de tinh correlation (n_agents={len(agents)}, "
            f"influence phan biet={len(set(influence_values))}). Can >= 3 agent voi "
            "influence score khac nhau."
        )
        return result

    pr, pp = pearsonr(influence_values, accuracy_values)
    sr, sp = spearmanr(influence_values, accuracy_values)
    result.update(pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp)
    return result
