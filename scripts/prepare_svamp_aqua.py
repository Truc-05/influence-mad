"""Normalize SVAMP (open-answer math, like GSM8K) and AQuA (5-way MC, like
CommonsenseQA) into the project's processed format: {question_id, question,
ground_truth}. For the E3b-v3 FROZEN external replication — new open-answer-reasoning
domains never seen by E3b. Run: python scripts/prepare_svamp_aqua.py
"""
import json, os, re, sys
from datasets import load_dataset

OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)


def prep_svamp():
    ds = load_dataset("ChilleD/SVAMP")
    rows = []
    for split in ds:
        for i, r in enumerate(ds[split]):
            q = (r.get("question_concat") or f"{r['Body']} {r['Question']}").strip()
            ans = r["Answer"]
            gt = str(int(ans)) if float(ans) == int(float(ans)) else str(ans)
            rows.append({"question_id": f"svamp_{r.get('ID', split+str(i))}",
                         "question": q, "ground_truth": gt})
    _write("svamp", rows)


def prep_aqua():
    ds = load_dataset("deepmind/aqua_rat", "raw")
    rows = []
    for split in ("test", "validation"):
        if split not in ds:
            continue
        for i, r in enumerate(ds[split]):
            opts = r["options"]
            if isinstance(opts, str):
                opts = eval(opts)                      # stored list-repr in some versions
            choices = ", ".join(re.sub(r"^([A-E])\)", r"\1) ", o.strip()) for o in opts)
            q = f"{r['question'].strip()}\nChoices: {choices}"
            rows.append({"question_id": f"aqua_{split}_{i}",
                         "question": q, "ground_truth": r["correct"].strip()})
    _write("aqua", rows)


def _write(name, rows):
    path = f"{OUT}/{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{name}] {len(rows)} questions -> {path}")
    print(f"  sample: {json.dumps(rows[0], ensure_ascii=False)[:160]}")


if __name__ == "__main__":
    prep_svamp()
    prep_aqua()
