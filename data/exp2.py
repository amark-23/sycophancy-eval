"""Experiment 2 data: GSM8K worked solutions, correct and with a corrupted final step.

Source: GSM8K (Cobbe et al. 2021), `openai/gsm8k` "main" config, test split; card license `mit`.
Each problem yields two versions of the same worked solution:
  correct    the reference solution, calculator annotations removed
  corrupted  the last calculation's result (and every later restatement of it) replaced by a wrong
             number (off by +-1 or +-10), so the final step is arithmetically wrong. Subtle.
  gross      same replacement but the wrong number is 10x the true value plus 3 (sign kept), so the
             error is large and visible. Used to check that ratings respond to quality when it is
             visible. Deterministic given the seed.
Items whose corruption cannot be done cleanly are dropped and counted. Dev/held-out split is by
hashing the item id (same scheme as Experiment 1). Question text is not committed, only ids.

Run:  python -m data.exp2 build
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from data.exp1 import _unit, content_hash, split_of

_ANNOT = re.compile(r"<<([^<>]*?)=([^<>=]*?)>>")
DELTAS = (-10, -1, 1, 10)


def load_gsm8k(split: str = "test") -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset("openai/gsm8k", "main", split=split))


def _to_int(s: str) -> int | None:
    s = s.strip().replace(",", "").lstrip("$")
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def parse_answer(answer: str) -> tuple[list[str], int | None]:
    """(step lines with the raw annotations kept, integer final answer or None)."""
    if "####" not in answer:
        return [], None
    body, final = answer.rsplit("####", 1)
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    return lines, _to_int(final)


def strip_annotations(line: str) -> str:
    return _ANNOT.sub("", line)


def _num_pattern(n: int) -> re.Pattern:
    plain = str(abs(n))
    forms = {plain, f"{abs(n):,}"}
    alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    sign = "-" if n < 0 else ""
    return re.compile(rf"(?<![\d,.]){sign}(?:{alt})(?![\d]|,\d|\.\d)")


def _fmt_like(old_match: str, new: int) -> str:
    return f"{new:,}" if "," in old_match else str(new)


def corrupt(lines: list[str], final: int, seed: int, key: str,
            kind: str = "subtle") -> tuple[list[str], int] | None:
    """Return (corrupted annotation-free lines, wrong final) or None if it cannot be done cleanly.
    kind: "subtle" (+-1 or +-10) or "gross" (10x the value plus 3, sign kept)."""
    idx = None
    for i, ln in enumerate(lines):
        for m in _ANNOT.finditer(ln):
            if _to_int(m.group(2)) == final:
                idx = i  # last line whose annotation yields the final answer
    if idx is None:
        return None
    if kind == "gross":
        wrong = final * 10 + (3 if final >= 0 else -3)
    else:
        delta = DELTAS[int(_unit(seed, "delta:" + key) * len(DELTAS))]
        wrong = final + delta
        if wrong == final or (final >= 0 and wrong < 0):
            wrong = final + abs(delta)
    pat = _num_pattern(final)
    out = []
    changed = False
    for i, ln in enumerate(lines):
        clean = strip_annotations(ln)
        if i >= idx:
            new = pat.sub(lambda m: _fmt_like(m.group(0), wrong), clean)
            changed = changed or new != clean
            clean = new
        out.append(clean)
    if not changed or not pat.search(strip_annotations(lines[idx])):
        return None
    return out, wrong


def render_solution(lines: list[str], final: int) -> str:
    return "\n".join(lines) + f"\nFinal answer: {final}"


def process(raw: list[dict], seed: int, dev_fraction: float, split_name: str = "test"):
    report = Counter()
    seen = set()
    recs = []
    for i, row in enumerate(raw):
        q = (row.get("question") or "").strip()
        if not q or q in seen:
            report["dropped_empty_or_duplicate"] += 1
            continue
        seen.add(q)
        lines, final = parse_answer(row.get("answer") or "")
        if final is None or not lines:
            report["dropped_unparseable_or_non_integer"] += 1
            continue
        rid = f"gsm8k-{split_name}-{i}"
        c = corrupt(lines, final, seed, rid)
        g = corrupt(lines, final, seed, rid, kind="gross")
        if c is None or g is None:
            report["dropped_corruption_not_clean"] += 1
            continue
        bad_lines, wrong = c
        gross_lines, gross = g
        good_lines = [strip_annotations(ln) for ln in lines]
        recs.append({"id": rid, "source": "gsm8k", "question": q,
                     "final_correct": final, "final_corrupted": wrong, "final_gross": gross,
                     "solution_correct": render_solution(good_lines, final),
                     "solution_corrupted": render_solution(bad_lines, wrong),
                     "solution_gross": render_solution(gross_lines, gross)})
        report["kept"] += 1
    out = {"dev": [], "heldout": []}
    for r in recs:
        out[split_of(r["id"], seed, dev_fraction)].append(r)
    return out, dict(report)


def build(cfg: dict, root: Path = Path(".")) -> dict:
    d = cfg["data"]
    split = d.get("gsm8k_split", "test")
    out, report = process(load_gsm8k(split), cfg["seed"], d["dev_fraction"], split)
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    for name, recs in out.items():
        with open(proc / f"exp2_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(recs, key=lambda r: r["id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"seed": cfg["seed"], "dev_fraction": d["dev_fraction"], "gsm8k_split": split,
                "report": report, "counts": {k: len(v) for k, v in out.items()},
                "content_sha256": {k: content_hash(v) for k, v in out.items()},
                "ids": {k: sorted(r["id"] for r in v) for k, v in out.items()}}
    mdir = root / "data" / "manifest"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "exp2_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["build"])
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    m = build(yaml.safe_load(open(args.config, encoding="utf-8")))
    print(json.dumps({"counts": m["counts"], "report": m["report"]}, indent=1))


if __name__ == "__main__":
    main()
