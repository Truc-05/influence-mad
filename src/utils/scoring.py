import re


def _normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s.]", "", text)
    return text.strip()


def _extract_number(text):
    matches = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])


def is_correct(predicted_answer, ground_truth, benchmark_name, correct_answers=None):
    if predicted_answer is None:
        return False

    if benchmark_name in ("gsm8k", "svamp"):
        pred_num = _extract_number(predicted_answer)
        gt_num = _extract_number(ground_truth)
        if pred_num is None or gt_num is None:
            return False
        return abs(pred_num - gt_num) < 1e-4

    if benchmark_name == "strategyqa":
        pred = _normalize_text(predicted_answer)
        gt = _normalize_text(ground_truth)
        pred_bool = pred.startswith("yes") or pred == "true"
        gt_bool = gt.startswith("yes") or gt == "true"
        return pred_bool == gt_bool

    if benchmark_name in ("commonsenseqa", "mmlu", "bbh", "aqua", "arc"):
        pred = _normalize_text(predicted_answer)
        gt = _normalize_text(ground_truth)
        pred_letter = pred[0] if pred else ""
        gt_letter = gt[0] if gt else ""
        return pred_letter == gt_letter or pred == gt

    if benchmark_name == "truthfulqa":
        pred = _normalize_text(predicted_answer)
        if not pred:
            return False
        candidates = [ground_truth] + list(correct_answers or [])
        for candidate in candidates:
            gt = _normalize_text(candidate)
            if not gt:
                continue
            if pred == gt:
                return True
            # Whole-phrase containment either direction, guarded by word
            # boundaries so short answers ("air", "no") match a verbose gold
            # phrase ("air is denser than water vapor") without matching
            # substrings inside longer words ("no" in "nowhere").
            if re.search(r"\b" + re.escape(pred) + r"\b", gt):
                return True
            if re.search(r"\b" + re.escape(gt) + r"\b", pred):
                return True
        return False

    pred = _normalize_text(predicted_answer)
    gt = _normalize_text(ground_truth)
    return pred == gt