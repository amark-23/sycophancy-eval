# Limitations

_Draft. Updated as the work proceeds; to be honest about what this eval does not show._

- Small models only (about 0.5B to 7B), so conclusions may not transfer to frontier models.
- Multiple-choice formats may not reflect free-form sycophancy.
- Public benchmark items may be contaminated in model training data.

## Experiment 1 (answer flip under pushback)

Written after the held-out run (2026-10-06). Numbers are from `results/pushback_summary.csv` and
`results/compare_shared.csv`.

- **Neutral control is degenerate.** When asked to answer again with no pushback, models repeat their
  own answer (S0 is about 0 for every model), so S - S0 is essentially S. The control rules out
  random flipping from resampling but does not isolate the effect of the social content of the message.
- **Suggestion versus sycophancy.** The pushback names an answer, so following it may be plain
  instruction-following. Evidence: models switch to a correct suggestion (U) about as often as, or more
  often than, to a wrong one (S - U is negative for the 1.5B, 3B and 7B, near zero for Llama-1B).
  Only the 7B shows clear discrimination (about 53% vs 88% for mild). A control where the user states a
  view and asks for an assessment, without telling the model what to answer, would separate the two.
  Not run.
- **Wording dependence.** Mild and strong differ in more than strength, and the two mild phrasings of
  the 7B differ by about 35 pp (35.4 vs 70.5 on its own known items; about 21 vs 58 on shared items).
  "The 7B resists mild pushback" holds for this wording pair only. Strength and wording are confounded;
  two phrasings per condition cannot separate them.
- **U is not evidence-based updating.** U measures willingness to move to a suggested correct answer on
  items the model gets wrong; the suggestion carries no evidence beyond the assertion.
- **Scale claims are not supported.** Three Qwen sizes in one family plus one Llama model. Mild-pushback
  switching falls with Qwen size on shared items (89.1, 80.3, 39.8), but strong pushback is not monotone
  (79.5, 92.9, 82.1), and family, training data and instruction tuning vary with size.
- **Item sets differ across models.** Each model's own known set differs, so own-set values are not
  directly comparable. Cross-model claims use the 390 items known by all four models, which are
  probably easier than average and are limited by Llama-1B's small known set (495 items). On shared items the
  7B's mild switch rate (39.8) is lower than on its own set (53.5).
- **Eligibility and selection.** Models were admitted by a clean-format rate (at least 95%) measured on dev.
  Llama-1B passed narrowly (96.6% dev, 96.2% held-out) and the 0.5B was excluded (89.3%), so the smallest
  model is not covered. Items unstable across baseline epochs are excluded by design.
- **Sampling.** Temperature 0.7, 3 epochs, no model seed (a fixed seed made epochs identical).
  Item-level means over 3 epochs are noisy for items near 50% stability; the bootstrap is over items and
  does not model within-item sampling noise separately.
- **Parse failures are excluded, not scored.** Rates are small (at most about 0.5% in any condition
  for the Qwen models; 13 of 2,748 neutral samples for Llama-1B) and listed in the metrics output.
- **Contamination and format.** ARC and MMLU are public; the models may have seen them. Multiple-choice
  letter answers may not reflect free-form sycophancy. Turn 1 is the model's own earlier answer written in
  a fixed format, which real conversations lack.
- **Dev was used to tune pushback wording** (before the freeze, hash 8ab5e7448ddd). Held-out results
  replicated dev within the CIs, but dev numbers are not reported as findings.

## Experiment 2 (feedback on worked math solutions)

Written at freeze (2026-10-08) from dev observations; held-out results will be checked against it.

- **Stated affect is weak evidence about quality.** People are more often unhappy with wrong work, so a model
  that lowers its rating after "I'm unhappy with it" may be using the cue as a hint, not deferring to the user.
  Dev shows the 7B's framing effect growing as the work gets worse (A 1.23, 1.77, 2.59 for correct, subtle,
  gross), which is consistent with either ceiling effects or hint-reading. A design that holds the cue
  uninformative cannot be guaranteed from the model's side.
- **Asymmetric effect.** Dev ratings drop sharply for "unhappy" and barely rise for "proud". The 3B and 7B rate
  correct work around 9 to 10 under neutral framing, so there is little room above neutral (ceiling).
- **The subtle error is mostly invisible.** Verdict accuracy is 0.51 to 0.55 on the subtle corruption and only
  0.56 to 0.68 on gross errors, so models are poor at checking these solutions at all. Q for the subtle version is
  at most 0.6 rating points. The gross version shows ratings respond to quality for the 3B and 7B (Q_gross about
  3 points) but barely for the 1.5B (0.87).
- **Direct rating, no reasoning.** Models answer in two lines with no reasoning step, which may overstate
  susceptibility relative to reasoning-first use.
- **Narrow task.** GSM8K arithmetic-ending solutions only; 146 of 1,319 problems were dropped because their final
  step has no calculator annotation (selection toward problems ending in an explicit calculation). Not essays,
  code or creative work.
- **Single family.** Only Qwen2.5 sizes (1.5B, 3B, 7B). Llama-3.2-1B was excluded because it returned a rating in
  only 16 to 38% of replies, so no second family is covered, and no scale claim is made.
- **Neutral wording differs.** The two neutral phrasings gave different mean ratings for the 3B on dev (8.1 vs 7.3,
  unpaired), so "proud - neutral" and "unhappy - neutral" depend somewhat on the neutral wording. A itself does not.
- **Sampling.** One epoch at temperature 0.7 on held-out, no seed; the bootstrap is over items and does not model
  within-item sampling noise separately.
- **Contamination.** GSM8K is public and widely used in training data.
