"""Build the turn-2 plan for each eligible model from its baseline log.

For every item the model's own baseline answers decide its class (known / known_wrong /
unstable, see analysis/known_items.py). The turn-1 transcript is the model's ACTUAL baseline
exchange (the prompt and one real reply), then the pushback is appended as a new user message.
So the pushback task only needs one `generate()` call per sample.

  known        turn 1 = first clean epoch (all epochs were correct)
  known_wrong  turn 1 = first clean epoch whose letter is the model's most frequent wrong letter
  unstable     excluded from the primary design

target_letter: the letter the pushback points at, fixed per item so conditions are paired.
  known        a random WRONG letter (seeded)
  known_wrong  the CORRECT letter
Items with no usable clean reply are skipped and counted.

Writes data/processed/pushback_plan_<split>_<model>.jsonl (git-ignored: it contains question
text). Run from the repo root after the baseline runs:  python -m analysis.pushback_plan
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from analysis.known_items import (CORRECT, classify, is_clean, latest_log_per_model, summarize,
                                  rows_from_log)
from tasks.prompts_exp1 import CONDITIONS_BY_ARM, prompt_hash, render

LETTERS = "ABCD"


def _bit(seed: int, key: str) -> int:
    """Stable 0/1 from (seed, key)."""
    return hashlib.sha256(f"{seed}:{key}".encode()).digest()[0] & 1


def plan_item(item: dict, model: str, seed: int):
    """One baseline item -> (list of plan records, skip_reason). Pure function.

    item: {id, user_prompt, choices, correct, source, subject,
           epochs: [{epoch, completion, letter, correct}]}
    """
    eps = sorted(item["epochs"], key=lambda e: e["epoch"])
    cls = classify([e["correct"] for e in eps])
    if cls == "unstable":
        return [], "unstable"
    correct = item["correct"]
    if cls == "known":
        pool = [e for e in eps if e["correct"]]
        initial = correct
        rng = random.Random(f"{seed}:{model}:{item['id']}:target")
        target = rng.choice([L for L in LETTERS if L != correct])
    else:
        wrong_letters = Counter(e["letter"] for e in eps if e["letter"] in LETTERS)
        if not wrong_letters:
            return [], "no_parsed_answer"
        initial = sorted(wrong_letters.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        pool = [e for e in eps if e["letter"] == initial]
        target = correct
    turn1 = next((e for e in pool if is_clean(e["completion"], e["letter"])), None)
    if turn1 is None:
        return [], "no_clean_turn1"

    records = []
    for cond in CONDITIONS_BY_ARM[cls]:
        phr = _bit(seed, f"{item['id']}:{cond}")
        needs_letter = cond not in ("neutral", "bare")
        text = render(cond, phr, target if needs_letter else None,
                      item["choices"][LETTERS.index(target)] if needs_letter else None)
        records.append({
            "sample_id": f"{item['id']}|{cond}",
            "item_id": item["id"], "condition": cond, "arm": cls, "phrasing": phr,
            "target_letter": target, "correct_letter": correct, "initial_letter": initial,
            "source": item["source"], "subject": item["subject"],
            "messages": [
                {"role": "user", "content": item["user_prompt"]},
                {"role": "assistant", "content": turn1["completion"]},
                {"role": "user", "content": text},
            ],
        })
    return records, cls


def items_from_log(log) -> list[dict]:
    """Group a baseline log's samples by item."""
    by: dict[str, dict] = {}
    for s in log.samples:
        score = next(iter(s.scores.values()))
        it = by.setdefault(str(s.id), {
            "id": str(s.id), "user_prompt": s.messages[0].text, "choices": list(s.choices),
            "correct": str(s.target if isinstance(s.target, str) else s.target[0]),
            "source": (s.metadata or {}).get("source", ""),
            "subject": (s.metadata or {}).get("subject", ""), "epochs": []})
        it["epochs"].append({"epoch": int(s.epoch), "completion": s.output.completion,
                             "letter": str(score.answer or "").strip(),
                             "correct": score.value == CORRECT})
    return list(by.values())


def slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def main() -> None:
    import yaml

    seed = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))["seed"]
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"pushback prompt hash: {prompt_hash()} (record this when the prompts are frozen)")
    for (model, split), log in latest_log_per_model().items():
        if not summarize(rows_from_log(log))["eligible"]:
            print(f"{model} [{split}]: NOT eligible, skipped")
            continue
        recs, skipped = [], Counter()
        for item in items_from_log(log):
            r, why = plan_item(item, model, seed)
            recs.extend(r)
            if not r:
                skipped[why] += 1
        path = out_dir / f"pushback_plan_{split}_{slug(model)}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in sorted(recs, key=lambda r: r["sample_id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per = Counter(r["condition"] for r in recs)
        print(f"{model} [{split}] -> {path.name}: {len(recs)} samples {dict(per)}; "
              f"skipped items {dict(skipped)}")


if __name__ == "__main__":
    main()
