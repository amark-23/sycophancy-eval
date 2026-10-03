"""Hello-world Inspect eval. Not part of the research results.

Purpose: check that Inspect can talk to a local Ollama model and show the four
core pieces in one place:
  Dataset  -> a list of Samples (input, choices, target letter)
  Solver   -> multiple_choice(): formats the question, calls the model
  Scorer   -> choice(): reads the model's letter and compares with target
  Task     -> ties them together; epochs=2 repeats every sample twice

Run (from the repo root, venv active, Ollama running):
    inspect eval tasks/hello.py --model ollama/qwen2.5:1.5b
Then view the log:
    inspect view
"""
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

# Three trivial hand-written items. Choices carry no letters: Inspect adds them.
SAMPLES = [
    Sample(
        input="What is the capital of France?",
        choices=["Berlin", "Paris", "Madrid", "Rome"],
        target="B",
    ),
    Sample(
        input="What is 7 * 8?",
        choices=["54", "56", "48", "64"],
        target="B",
    ),
    Sample(
        input="Which planet is closest to the Sun?",
        choices=["Venus", "Earth", "Mercury", "Mars"],
        target="C",
    ),
]


@task
def hello():
    return Task(
        dataset=MemoryDataset(SAMPLES),
        solver=multiple_choice(),
        scorer=choice(),
        epochs=2,
    )
