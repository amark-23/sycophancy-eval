"""Baseline accuracy: plain multiple choice, NO pushback.

Purpose: measure how often each model answers each item correctly, k times per item
(epochs), so we can see how many "known" items (correct on all k samples) each model
has. That number sets the Experiment 1 sample sizes. Dev split only: the held-out
split is not touched until prompts are frozen.

Build the data first:  python -m data.exp1 build

Run (repo root, venv active, Ollama running), small check first:
    inspect eval tasks/baseline.py --model ollama/qwen2.5:1.5b -T split=dev ^
        --epochs 3 --temperature 0.7 --seed 0 --limit 50 --log-dir results/logs
Drop --limit for the full dev split.
(In PowerShell use ` as the line-continuation character instead of ^, or put it on one line.)
"""
import json
import sys
from pathlib import Path

# `inspect eval` loads this file without the repo root on sys.path, so add it before
# importing our own packages (data/). Resolved from this file, not the working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from data.exp1 import to_samples


def load_split(split: str):
    path = ROOT / "data" / "processed" / f"exp1_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m data.exp1 build")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@task
def baseline(split: str = "dev"):
    return Task(
        dataset=MemoryDataset(to_samples(load_split(split)), name=f"exp1_{split}"),
        solver=multiple_choice(),
        scorer=choice(),
    )
