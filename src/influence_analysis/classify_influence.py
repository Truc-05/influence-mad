import numpy as np

from src.influence_analysis.correlation import accuracy_per_agent


def classify_agents(debate_logs, influence_scores, benchmark_name, agent_names=None):
    """Phân loại Good / Bad / Low influence dựa trên influence score và accuracy.

    - good_influence: influence cao VÀ accuracy cao (ảnh hưởng tốt, nên tin).
    - bad_influence:  influence cao NHƯNG accuracy thấp (kéo nhóm đi sai).
    - low_influence:  influence thấp.
    Ngưỡng "cao/thấp" lấy theo median trên tập agent.
    """
    if agent_names is None:
        agent_names = list(influence_scores.keys())

    acc = accuracy_per_agent(debate_logs, benchmark_name, agent_names)

    influence_values = [influence_scores.get(n, 0.0) for n in agent_names]
    accuracy_values = [acc.get(n, 0.0) for n in agent_names]
    median_influence = float(np.median(influence_values))
    median_accuracy = float(np.median(accuracy_values))

    classification = {}
    for name in agent_names:
        high_influence = influence_scores.get(name, 0.0) >= median_influence
        high_accuracy = acc.get(name, 0.0) >= median_accuracy
        if high_influence and high_accuracy:
            classification[name] = "good_influence"
        elif high_influence and not high_accuracy:
            classification[name] = "bad_influence"
        else:
            classification[name] = "low_influence"

    return classification, acc
