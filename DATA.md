# Data sources and licenses

Checked on 2026-10-04 against the Hugging Face dataset cards only. Not legal advice.

| Dataset | Source | License on card | Use here |
|---|---|---|---|
| ARC-Challenge | `allenai/ai2_arc` (Hugging Face) | CC BY-SA 4.0 | Experiment 1 items (4-option only) |
| MMLU | `cais/mmlu` (Hugging Face) | MIT | Experiment 1 items (subjects TBD) |

## Redistribution rule
Question data is **not committed** to this repo. The loader downloads from Hugging Face,
and the repo stores only item IDs and the split seed, so the exact dev and held-out
splits can be rebuilt. This avoids redistributing ARC content, which would otherwise
require releasing the copy under CC BY-SA 4.0 with attribution.
Our own code and prompts are covered by this repo's MIT license.

## Citations
- Clark et al., "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", arXiv:1803.05457 (2018).
- Hendrycks et al., "Measuring Massive Multitask Language Understanding", ICLR 2021, arXiv:2009.03300.

## Caveats
- MMLU questions were compiled from other sources (exams, textbooks). The card's MIT label
  was not checked against those underlying sources.
- Contamination has not been checked. Both sets are likely in model training data
  (to be noted in LIMITATIONS.md).
