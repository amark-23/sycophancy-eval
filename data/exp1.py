"""Experiment 1 dataset: 4-option multiple-choice items from ARC-Challenge and MMLU.

Pipeline (all deterministic given the seed):
  1. load raw records from Hugging Face            (load_arc / load_mmlu)
  2. normalize to one schema                        (normalize_arc / normalize_mmlu)
  3. clean: drop bad items and count why            (clean)
  4. shuffle each item's choices with a per-item seed so the correct letter is not
     always the same, keeping the answer letter correct  (shuffle_item)
  5. assign dev / held-out by hashing the item id   (split_of)
  6. write a manifest of ids + a content hash       (build)

Question text is NOT committed to git (see DATA.md): processed files go to
data/processed/ (git-ignored), and only the id manifest is tracked.

Record schema (dict):
  id        "arc:<ARC id>" or "mmlu:<subject>:<row index in the test split>"
  source    "arc" | "mmlu"
  subject   subject name ("arc" for ARC)
  question  str
  choices   list of 4 str, no letters
  answer    "A".."D"  (letter of the correct choice in `choices`)

Run from the repo root:  python -m data.exp1 build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

LETTERS = "ABCD"

# Options that refer to other options ("All of the above", "Both A and B") break when the
# choices are shuffled, so such items are dropped. The phrase pattern ignores case; the
# letter pattern is case-sensitive so ordinary words like "and a" are not caught.
_PHRASE = re.compile(r"\b(all|none|neither|both)\s+of\s+(the\s+)?(above|these|them)\b", re.IGNORECASE)
_LETTER_REF = re.compile(r"\b(?:[Bb]oth|and|or)\s+[A-D]\b|\b[A-D]\s+(?:and|or)\s+[A-D]\b")


def _positional(choice: str) -> bool:
    return bool(_PHRASE.search(choice) or _LETTER_REF.search(choice))


# ----------------------------------------------------------------------------- loading
def load_arc(split: str = "test") -> list[dict]:
    """Raw ARC-Challenge rows from Hugging Face (needs the `datasets` package)."""
    from datasets import load_dataset

    return list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split))


def load_mmlu(subject: str, split: str = "test") -> list[dict]:
    """Raw MMLU rows for one subject, each tagged with its row index (for stable ids)."""
    from datasets import load_dataset

    rows = list(load_dataset("cais/mmlu", subject, split=split))
    for i, r in enumerate(rows):
        r["_index"] = i
    return rows


# -------------------------------------------------------------------------- normalizing
def normalize_arc(row: dict) -> dict | None:
    """ARC row -> schema. Returns None unless the item has exactly 4 choices.

    ARC labels are mostly A-D but some items use 1-4; we go by position, not label.
    """
    texts = list(row["choices"]["text"])
    labels = list(row["choices"]["label"])
    if len(texts) != 4:
        return None
    key = str(row["answerKey"])
    if key not in labels:
        return None
    return {
        "id": f"arc:{row['id']}",
        "source": "arc",
        "subject": "arc",
        "question": str(row["question"]).strip(),
        "choices": [str(t).strip() for t in texts],
        "answer": LETTERS[labels.index(key)],
    }


def normalize_mmlu(row: dict) -> dict | None:
    """MMLU row -> schema. `answer` in MMLU is an int index 0-3."""
    choices = list(row["choices"])
    if len(choices) != 4 or int(row["answer"]) not in range(4):
        return None
    return {
        "id": f"mmlu:{row['subject']}:{row['_index']}",
        "source": "mmlu",
        "subject": str(row["subject"]),
        "question": str(row["question"]).strip(),
        "choices": [str(c).strip() for c in choices],
        "answer": LETTERS[int(row["answer"])],
    }


# ---------------------------------------------------------------------------- cleaning
def clean(records: list[dict]) -> tuple[list[dict], Counter]:
    """Drop items that would be ambiguous or broken after shuffling. Returns (kept, reasons).

    Reasons: empty_text, duplicate_choices, positional_choice, duplicate_question.
    """
    kept, why, seen = [], Counter(), set()
    for r in records:
        if not r["question"] or any(not c for c in r["choices"]):
            why["empty_text"] += 1
        elif len({c.lower() for c in r["choices"]}) < 4:
            why["duplicate_choices"] += 1
        elif any(_positional(c) for c in r["choices"]):
            why["positional_choice"] += 1
        elif r["question"].lower() in seen:
            why["duplicate_question"] += 1
        else:
            seen.add(r["question"].lower())
            kept.append(r)
    return kept, why


# ------------------------------------------------------------------- shuffle and split
def _unit(seed: int, key: str) -> float:
    """Stable pseudo-random number in [0, 1) from (seed, key); independent of data order."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def shuffle_item(rec: dict, seed: int) -> dict:
    """Permute the choices with a per-item seed and move the answer letter with them."""
    correct_text = rec["choices"][LETTERS.index(rec["answer"])]
    rng = random.Random(f"{seed}:{rec['id']}")
    order = list(range(4))
    rng.shuffle(order)
    choices = [rec["choices"][i] for i in order]
    out = dict(rec, choices=choices, answer=LETTERS[choices.index(correct_text)])
    return out


def split_of(rec_id: str, seed: int, dev_fraction: float) -> str:
    """'dev' or 'heldout', decided by hashing the id (so adding items never moves others)."""
    return "dev" if _unit(seed, "split:" + rec_id) < dev_fraction else "heldout"


# -------------------------------------------------------------------------------- build
def content_hash(records: list[dict]) -> str:
    """Hash of the processed records, so a rebuild can be checked against the manifest."""
    blob = json.dumps(sorted(records, key=lambda r: r["id"]), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def process(raw_arc: list[dict], raw_mmlu: list[dict], seed: int, dev_fraction: float):
    """Normalize, clean, shuffle and split. Pure function: no network, no disk."""
    report: dict = {}
    sources = {
        "arc": [n for r in raw_arc if (n := normalize_arc(r))],
        "mmlu": [n for r in raw_mmlu if (n := normalize_mmlu(r))],
    }
    raw_counts = {"arc": len(raw_arc), "mmlu": len(raw_mmlu)}
    out: dict[str, list[dict]] = {"dev": [], "heldout": []}
    for name, recs in sources.items():
        kept, why = clean(recs)
        report[name] = {
            "raw_rows": raw_counts[name],
            "four_option": len(recs),
            "kept": len(kept),
            "dropped": dict(why),
        }
        for r in kept:
            out[split_of(r["id"], seed, dev_fraction)].append(shuffle_item(r, seed))
    return out, report


def to_samples(records: list[dict]):
    """Records -> Inspect Samples (for multiple_choice() + choice()). Imported lazily."""
    from inspect_ai.dataset import Sample

    return [
        Sample(
            id=r["id"],
            input=r["question"],
            choices=r["choices"],
            target=r["answer"],
            metadata={"source": r["source"], "subject": r["subject"]},
        )
        for r in records
    ]


def build(cfg: dict, root: Path = Path(".")) -> dict:
    """Download, process, write data/processed/*.jsonl (ignored) and the id manifest (tracked)."""
    d = cfg["data"]
    raw_arc = load_arc(d.get("arc_split", "test"))
    raw_mmlu = [r for s in d["mmlu_subjects"] for r in load_mmlu(s, d.get("mmlu_split", "test"))]
    out, report = process(raw_arc, raw_mmlu, cfg["seed"], d["dev_fraction"])

    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    for split, recs in out.items():
        with open(proc / f"exp1_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(recs, key=lambda r: r["id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": cfg["seed"],
        "dev_fraction": d["dev_fraction"],
        "mmlu_subjects": d["mmlu_subjects"],
        "report": report,
        "counts": {k: len(v) for k, v in out.items()},
        "content_sha256": {k: content_hash(v) for k, v in out.items()},
        "ids": {k: sorted(r["id"] for r in v) for k, v in out.items()},
    }
    mdir = root / "data" / "manifest"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "exp1_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["build"])
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m = build(cfg)
    print(json.dumps({"counts": m["counts"], "report": m["report"]}, indent=1))


if __name__ == "__main__":
    main()
