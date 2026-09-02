from collections import Counter


def majority_voting(candidate_answers):
    normalized = [a.strip().lower() for a in candidate_answers.values()]
    counts = Counter(normalized)
    winner = counts.most_common(1)[0][0]
    for name, answer in candidate_answers.items():
        if answer.strip().lower() == winner:
            return answer
    return list(candidate_answers.values())[0]


def influence_based_voting(candidate_answers, influence_scores):
    weighted = {}
    for name, answer in candidate_answers.items():
        key = answer.strip().lower()
        weighted[key] = weighted.get(key, 0.0) + influence_scores.get(name, 0.0)

    winner_key = max(weighted, key=weighted.get)
    for name, answer in candidate_answers.items():
        if answer.strip().lower() == winner_key:
            return answer
    return list(candidate_answers.values())[0]


def influence_aware_voting(candidate_answers, influence_scores, agent_classification):
    adjusted_scores = {}
    for name, score in influence_scores.items():
        label = agent_classification.get(name, "low_influence")
        if label == "bad_influence":
            adjusted_scores[name] = score * 0.3
        elif label == "good_influence":
            adjusted_scores[name] = score * 1.2
        else:
            adjusted_scores[name] = score

    return influence_based_voting(candidate_answers, adjusted_scores)
