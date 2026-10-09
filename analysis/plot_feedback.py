"""Figure for Experiment 2 from results/feedback_summary.csv (no log reading, so it runs instantly).

Left: framing effect (proud - unhappy) on the primary pooled items and on the gross-error version.
Right: quality effect under neutral framing (correct minus subtle error, correct minus gross error).
Both in rating points (1-10), 95% bootstrap CI over items.

Run:  python -m analysis.plot_feedback     -> results/fig_exp2.png
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path="results/feedback_summary.csv", split="heldout"):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["split"] == split:
            d[(r["model"], r["metric"])] = (float(r["estimate"]), float(r["ci_lo"]), float(r["ci_hi"]))
    return d


def main() -> None:
    d = load()
    models = sorted({m for m, _ in d}, key=lambda m: float(m.split(":")[1].rstrip("b")))
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    panels = (("Framing effect: proud - unhappy", (("A", "primary (correct + subtle)"), ("A_gross", "gross-error version"))),
              ("Quality effect (neutral framing)", (("Q", "correct - subtle error"), ("Q_gross", "correct - gross error"))))
    for ax, (title, series) in zip(axes, panels):
        for k, (metric, label) in enumerate(series):
            vals = [d[(m, metric)] for m in models]
            est = np.array([v[0] for v in vals])
            ax.errorbar(x + (k - 0.5) * 0.15, est,
                        yerr=[est - np.array([v[1] for v in vals]), np.array([v[2] for v in vals]) - est],
                        fmt="o" if k == 0 else "s", capsize=3, label=label)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x, [m.replace("ollama/", "") for m in models])
        ax.set_title(title)
        ax.set_ylabel("rating points (1-10)")
        ax.legend(fontsize=8)
    fig.suptitle("Experiment 2, held-out (814 items, 1 epoch); 95% bootstrap CI over items")
    fig.tight_layout()
    fig.savefig("results/fig_exp2.png", dpi=150)


if __name__ == "__main__":
    main()
