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
_To be written at pre-registration._

## Experiment 1 data (see data/exp1.py, DATA.md)
- Sources: ARC-Challenge (test split, 4-option items only) and selected MMLU subjects.
- Cleaning drops: empty text, duplicate choices (ignoring case), duplicate questions, and any item
  with an option that refers to other options ("All of the above", "Both A and B"), because choices are shuffled.
  Drop counts per reason are written to the manifest.
- Each item's choices are shuffled with a per-item seed, and the answer letter moves with them.
- Dev / held-out split: decided by hashing the item id with the seed, so it does not depend on data order.
- Question text is not committed. `data/manifest/exp1_manifest.json` stores ids, counts and a content hash
  so a rebuild can be checked.
