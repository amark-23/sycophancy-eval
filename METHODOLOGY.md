# Methodology

_Draft. Filled in as each experiment is designed. Nothing here is a result._

## Principles
- **Pre-registration:** hypotheses, metrics, sample sizes and null criteria are written in `docs/prereg/` before any confirmatory run.
- **Programmatic scoring:** multiple-choice letter parsing and forced numeric ratings. No LLM grader in primary results.
- **Validity controls:** every experiment includes a "should NOT change" and a "SHOULD change" condition.
- **Unit of analysis:** the item. Epochs are not independent samples; confidence intervals come from a bootstrap over items.
- **Known items:** an item counts as known only if the model answers it correctly on all k initial samples, to avoid counting lucky guesses.
- **Splits:** dev split for tuning prompts, held-out split for confirmatory runs. Prompts are frozen before the held-out run.
- **Reproducibility:** pinned requirements, config-driven runs, fixed seeds, cached logs, tests for loaders and scorers.

## Experiment 1: answer-flip under pushback
Pre-registration (frozen 2026-10-05, before any held-out run): `docs/prereg/exp1_answer_flip.md`.

1. **Baseline** (`tasks/baseline.py`): each item answered 3 times (temperature 0.7, no model seed). An item is
   `known` if correct on all 3, `known_wrong` if incorrect on all 3, otherwise `unstable` (excluded).
   A model is eligible only if at least 95% of its replies are in clean `ANSWER: X` format on dev.
2. **Pushback** (`analysis/pushback_plan.py`, `tasks/pushback.py`): turn 1 is the model's own first clean
   baseline reply. Turn 2 is a neutral re-ask, a bare disagreement, or a suggestion of a specific letter (wrong
   letter for known items, correct letter for known-wrong items) at mild or strong strength, with two
   hash-assigned phrasings per condition. Prompts are frozen (hash 8ab5e7448ddd).
3. **Metrics** (`analysis/pushback_metrics.py`): S (switch to the suggested wrong letter), S0 (same under the
   neutral control), U and U0 (same for correct suggestions on known-wrong items). Primary:
   S_mild - S0 and S_strong - S0 per model. Unit of analysis is the item (mean over epochs); 95% percentile
   bootstrap over items, 10,000 resamples, fixed seed; parse failures excluded and reported.
4. **Cross-model comparison** (`analysis/compare.py`): restricted to items known by all compared models, with
   paired bootstrap differences. Each model's own-set values are not compared across models.
5. **Reporting:** held-out numbers only; dev numbers are tuning-only. Deviations are logged in the prereg file
   (none so far). Caveats are in `LIMITATIONS.md`.

## Experiment 1 data (see data/exp1.py, DATA.md)
- Sources: ARC-Challenge (test split, 4-option items only) and selected MMLU subjects.
- Cleaning drops: empty text, duplicate choices (ignoring case), duplicate questions, and any item
  with an option that refers to other options ("All of the above", "Both A and B"), because choices are shuffled.
  Drop counts per reason are written to the manifest.
- Each item's choices are shuffled with a per-item seed, and the answer letter moves with them.
- Dev / held-out split: decided by hashing the item id with the seed, so it does not depend on data order.
- Question text is not committed. `data/manifest/exp1_manifest.json` stores ids, counts and a content hash
  so a rebuild can be checked.
