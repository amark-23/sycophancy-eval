# sycophancy-eval

A reproducible, statistically careful evaluation of **sycophancy** in small open language models, built with [Inspect](https://inspect.aisi.org.uk/).

**Status:** scaffold only. No results yet.

## Research question
When a user pushes back, states a view, or presents work as their own, how much do small LLMs shift their answers, and how does this vary with model size? The eval separates sycophancy from general instability by including conditions where the model should change its answer and conditions where it should not.

## Experiments (planned order)
1. Answer-flip under pushback (multiple choice, verifiable answers, with controls)
2. Feedback sycophancy (same work framed as mine / someone else's / with stated enthusiasm or dislike)
3. Opinion and identity mirroring (factually answerable questions only)
4. Stretch: scaling within one model family, optional linear probe

## Layout
| Path | Purpose |
|---|---|
| `tasks/` | Inspect tasks |
| `data/` | datasets (`raw/` sources, `processed/` cleaned splits) with provenance |
| `scorers/` | programmatic scorers |
| `analysis/` | bootstrap CIs, paired comparisons, figures |
| `tests/` | tests for data loaders and scorers |
| `results/` | tables, figures, summaries (raw logs stay local) |
| `configs/` | run configuration |
| `docs/prereg/` | pre-registrations, written before each experiment runs |

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Then install [Ollama](https://ollama.com) and pull the models listed in `configs/default.yaml`.

## Reproduce
```
python reproduce.py
```

See `METHODOLOGY.md` and `LIMITATIONS.md`.
