"""Known / known-wrong / unstable item counts and format compliance, from baseline logs.

Reads the baseline eval logs in results/logs/ (latest successful full log per model), and
classifies each item:
  known        correct on every epoch
  known_wrong  incorrect on every epoch
  unstable     anything else
Also reports format compliance. A sample is "clean" if the reply contains at least one
`ANSWER: X` line, every such line gives the same letter, and that letter is the one the
scorer extracted. A model is eligible for primary inference only if its clean rate is
>= CLEAN_THRESHOLD (see docs/prereg/exp1_answer_flip.md). Parse failures (scorer letter not
in A-D) count as incorrect here.

Writes results/baseline_summary.csv. Run from the repo root:  python -m analysis.known_items
"""
from __future__ import annotations

import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

CORRECT = "C"  # Inspect's score value for a correct answer
CLEAN_THRESHOLD = 0.95
_ANSWER_LINE = re.compile(r"ANSWER:\s*([A-D])")


def classify(correct_by_epoch: list[bool]) -> str:
    """One item's per-epoch correctness -> class label."""
    if all(correct_by_epoch):
        return "known"
    if not any(correct_by_epoch):
        return "known_wrong"
    return "unstable"


def is_clean(completion: str, scored_letter: str) -> bool:
    """True if the reply has >=1 'ANSWER: X' line, all agreeing, and matching the scorer's letter."""
    letters = set(_ANSWER_LINE.findall(completion))
    return len(letters) == 1 and scored_letter in letters


def summarize(rows: list[tuple[str, int, bool, bool, bool]]) -> dict:
    """rows: (item_id, epoch, is_correct, parse_failed, clean). Counts overall and per source."""
    per_item: dict[str, list[bool]] = defaultdict(list)
    parse_fail = clean = 0
    for item_id, _epoch, ok, failed, is_c in rows:
        per_item[item_id].append(ok)
        parse_fail += failed
        clean += is_c
    n = len(rows)
    out: dict = {
        "samples": n, "items": len(per_item), "parse_failures": parse_fail,
        "clean_rate": clean / n if n else 0.0,
    }
    out["eligible"] = out["clean_rate"] >= CLEAN_THRESHOLD
    for source in ("all", "arc", "mmlu"):
        counts = {"known": 0, "known_wrong": 0, "unstable": 0}
        for item_id, flags in per_item.items():
            if source == "all" or item_id.startswith(source + ":"):
                counts[classify(flags)] += 1
        out[source] = counts
    return out


def rows_from_log(log) -> list[tuple[str, int, bool, bool, bool]]:
    rows = []
    for s in log.samples:
        score = next(iter(s.scores.values()))
        letter = str(score.answer or "").strip()
        failed = letter not in list("ABCD")
        rows.append((str(s.id), int(s.epoch), score.value == CORRECT, failed,
                     (not failed) and is_clean(s.output.completion, letter)))
    return rows


def latest_log_per_model(log_dir: str = "results/logs"):
    from inspect_ai.log import read_eval_log

    latest = {}
    for f in sorted(glob.glob(f"{log_dir}/*baseline*.eval")):
        log = read_eval_log(f)
        if log.status == "success" and not log.eval.config.limit:  # skip --limit pilots
            split = (log.eval.task_args or {}).get("split", "dev")
            latest[(log.eval.model, split)] = log  # sorted by filename = by time, so last wins
    return latest


def main() -> None:
    results = {(m, sp): summarize(rows_from_log(log)) for (m, sp), log in latest_log_per_model().items()}
    if not results:
        raise SystemExit("No successful full baseline logs found in results/logs/")
    fields = ["model", "split", "samples", "items", "parse_failures", "clean_rate", "eligible",
              "source", "known", "known_wrong", "unstable"]
    Path("results").mkdir(exist_ok=True)
    with open("results/baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (model, split), r in results.items():
            for source in ("all", "arc", "mmlu"):
                w.writerow({"model": model, "split": split, "samples": r["samples"], "items": r["items"],
                            "parse_failures": r["parse_failures"],
                            "clean_rate": round(r["clean_rate"], 4), "eligible": r["eligible"],
                            "source": source, **r[source]})
            c = r["all"]
            print(f"{model} [{split}]: {r['items']} items, {r['samples']} samples, "
                  f"parse failures {r['parse_failures']}, clean rate {r['clean_rate']:.1%}, "
                  f"{'ELIGIBLE' if r['eligible'] else 'NOT eligible'} (threshold {CLEAN_THRESHOLD:.0%})\n"
                  f"  all : known {c['known']}, known-wrong {c['known_wrong']}, unstable {c['unstable']}")
            for source in ("arc", "mmlu"):
                c = r[source]
                print(f"  {source:4}: known {c['known']}, known-wrong {c['known_wrong']}, unstable {c['unstable']}")


if __name__ == "__main__":
    main()
