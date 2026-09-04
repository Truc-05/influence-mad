"""Normalize MMLU (4-way MC, factual recall) and ARC-Challenge (4-way MC science)
into the project's processed format {question_id, question, ground_truth}, mirroring
the AQuA/CommonsenseQA convention: choices embedded as "\\nChoices: A) ..., B) ...",
ground_truth = the answer LETTER. For the E25 breadth extension (prereg
paper/E25_newbench_prereg.md). Deterministic seed-42 shuffle so the first-N slice
(gen_cell reads [:n]) is a representative mix, not one MMLU subject.

Run (offline, uses local HF cache):
  HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python scripts/prepare_mmlu_arc.py
"""
import json, os, random
from datasets import load_dataset

OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)
_LETTERS = ["A", "B", "C", "D", "E"]


def _fmt(question, options):
    """options = list of choice strings, in order -> relabelled A) B) C)..."""
    choices = ", ".join(f"{_LETTERS[i]}) {o.strip()}" for i, o in enumerate(options))
    return f"{question.strip()}\nChoices: {choices}"


def prep_mmlu(limit=500, seed=42):
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i, r in enumerate(ds):
        opts = r["choices"]                       # list[str], 4 options
        ans = int(r["answer"])                    # 0-based index
        if not (0 <= ans < len(opts)):
            continue
        rows.append({"question_id": f"mmlu_{i}",
                     "question": _fmt(r["question"], opts),
                     "ground_truth": _LETTERS[ans]})
    random.Random(seed).shuffle(rows)             # representative first-N slice
    _write("mmlu", rows[:limit])


def prep_arc(limit=500, seed=42):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i, r in enumerate(ds):
        texts = r["choices"]["text"]              # list[str]
        labels = r["choices"]["label"]            # e.g. ["A","B","C","D"] or ["1","2","3","4"]
        key = str(r["answerKey"]).strip()
        if key not in labels:                     # skip malformed
            continue
        pos = labels.index(key)                   # position -> relabel to a LETTER
        if pos >= len(_LETTERS):
            continue
        rows.append({"question_id": f"arc_{r.get('id', i)}",
                     "question": _fmt(r["question"], texts),
                     "ground_truth": _LETTERS[pos]})
    random.Random(seed).shuffle(rows)
    _write("arc", rows[:limit])


def _write(name, rows):
    path = f"{OUT}/{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{name}] {len(rows)} questions -> {path}")
    print(f"  sample: {json.dumps(rows[0], ensure_ascii=False)[:220]}")


if __name__ == "__main__":
    prep_mmlu()
    prep_arc()
