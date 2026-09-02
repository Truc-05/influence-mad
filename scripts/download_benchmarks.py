import os
import json
import random
import re

from datasets import load_dataset


def _save(records, benchmark_name):
    os.makedirs("data/processed", exist_ok=True)
    path = f"data/processed/{benchmark_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{benchmark_name}: {len(records)} samples -> {path}")


def prepare_gsm8k(limit=500, seed=42):
    ds = load_dataset("gsm8k", "main", split="test")
    records = []
    for i, item in enumerate(ds):
        answer = item["answer"].split("####")[-1].strip()
        records.append({
            "question_id": f"gsm8k_{i}",
            "question": item["question"],
            "ground_truth": answer,
        })
    random.Random(seed).shuffle(records)
    _save(records[:limit], "gsm8k")


def prepare_strategyqa(limit=500, seed=42):
    ds = load_dataset("ChilleD/StrategyQA", split="test")
    records = []
    for i, item in enumerate(ds):
        records.append({
            "question_id": f"strategyqa_{i}",
            "question": item["question"],
            "ground_truth": "yes" if item["answer"] else "no",
        })
    random.Random(seed).shuffle(records)
    _save(records[:limit], "strategyqa")


def prepare_commonsenseqa(limit=500, seed=42):
    ds = load_dataset("tau/commonsense_qa", split="validation")
    records = []
    for i, item in enumerate(ds):
        choices_text = ", ".join(
            f"{label}) {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])
        )
        question = f"{item['question']}\nChoices: {choices_text}"
        records.append({
            "question_id": f"commonsenseqa_{i}",
            "question": question,
            "ground_truth": item["answerKey"],
        })
    random.Random(seed).shuffle(records)
    _save(records[:limit], "commonsenseqa")


def prepare_truthfulqa(limit=500, seed=42):
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    records = []
    for i, item in enumerate(ds):
        records.append({
            "question_id": f"truthfulqa_{i}",
            "question": item["question"],
            "ground_truth": item["best_answer"],
            "correct_answers": item["correct_answers"],
        })
    random.Random(seed).shuffle(records)
    _save(records[:limit], "truthfulqa")


# BBH gồm 27 subtask với format target khác nhau (letter, True/False, số, chuỗi...).
# Để dùng chung bộ chấm letter-based (nhánh "bbh" trong scoring.py, giống
# commonsenseqa), ta chỉ lấy các subtask dạng multiple-choice có target "(X)".
# input của BBH đã kèm sẵn phần "Options: (A) ... (B) ...", nên agent trả lời
# bằng 1 chữ cái y như commonsenseqa.
_BBH_MCQ_TASKS = [
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
]

_BBH_LETTER_RE = re.compile(r"^\(([A-Z])\)$")


def prepare_bbh(limit=500, seed=42, tasks=None):
    tasks = tasks or _BBH_MCQ_TASKS
    records = []
    for task in tasks:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception as e:
            print(f"  [skip] bbh/{task}: {e}")
            continue
        for i, item in enumerate(ds):
            target = str(item["target"]).strip()
            m = _BBH_LETTER_RE.match(target)
            if not m:
                continue  # bỏ item không phải multiple-choice "(X)"
            records.append({
                "question_id": f"bbh_{task}_{i}",
                "question": item["input"],
                "ground_truth": m.group(1),  # lưu chữ cái trần: "A", "B", ...
            })
    random.Random(seed).shuffle(records)
    _save(records[:limit], "bbh")


if __name__ == "__main__":
    prepare_gsm8k()
    prepare_strategyqa()
    prepare_commonsenseqa()
    prepare_truthfulqa()
    prepare_bbh()