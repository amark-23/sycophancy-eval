"""Cross-model comparison on items that ALL compared models know (held-out split).

Each model's own S is computed on its own known items, so those numbers are not directly
comparable across models (a stronger model knows more, and harder, items). Here every model is
scored on the same items: those known by all models (and parsed in both wrong_* conditions).
Paired differences between models use the same item-level bootstrap as pushback_metrics.

Run after the held-out pushback runs:  python -m analysis.compare
Writes results/compare_shared.csv, results/fig_primary.png, results/fig_phrasing.png.
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

from analysis.pushback_metrics import (item_means, mean_ci, paired_diff_ci, rows_from_log,
                                       SEED)

CONDS = ("wrong_mild", "wrong_strong")


def switch_means(rows, phrasing=None) -> dict[str, dict[str, float]]:
    """condition -> item_id -> mean P(final == target), optionally for one phrasing."""
    sub = rows if phrasing is None else [r for r in rows if r["phrasing"] == phrasing]
    return {c: item_means(sub, c, "known", lambda r: r["final"] == r["target"]) for c in CONDS}


def shared_items(per_model: dict[str, dict[str, dict[str, float]]]) -> set[str]:
    """Items present in every model's wrong_mild AND wrong_strong item means."""
    sets = [set(m[c]) for m in per_model.values() for c in CONDS]
    return set.intersection(*sets) if sets else set()


def restrict(d: dict[str, float], items: set[str]) -> dict[str, float]:
    return {i: v for i, v in d.items() if i in items}


def heldout_logs(log_dir: str = "results/logs") -> dict:
    """Latest successful full held-out pushback log per model. Headers are read first so that
    only the logs actually needed are loaded in full (the logs are large)."""
    import glob
    from inspect_ai.log import read_eval_log

    latest = {}
    for f in sorted(glob.glob(f"{log_dir}/*pushback*.eval")):
        h = read_eval_log(f, header_only=True)
        plan = str((h.eval.task_args or {}).get("plan", ""))
        if h.status == "success" and not h.eval.config.limit and "heldout" in plan:
            latest[h.eval.model] = f
    return {m: read_eval_log(f) for m, f in latest.items()}


def main() -> None:
    logs = heldout_logs()
    logs = dict(sorted(logs.items(), key=lambda kv: ("llama" in kv[0], kv[0])))  # Qwen by size, then Llama
    if len(logs) < 2:
        raise SystemExit("Need held-out pushback logs for at least two models.")
    rows = {m: rows_from_log(l) for m, l in logs.items()}
    per_model = {m: switch_means(r) for m, r in rows.items()}
    shared = shared_items(per_model)
    print(f"{len(logs)} models; {len(shared)} items known by all (each model's own known set: "
          f"{ {m: len(v['wrong_mild']) for m, v in per_model.items()} })")
    rng = np.random.default_rng(SEED)
    out = []
    for m in logs:
        for c in CONDS:
            own = mean_ci(per_model[m][c], rng)
            sh = mean_ci(restrict(per_model[m][c], shared), rng)
            out.append(dict(kind="level", model=m, other="", cond=c, phrasing="all",
                            est=sh["est"], lo=sh["lo"], hi=sh["hi"], n=sh["n_items"],
                            own_est=own["est"], own_n=own["n_items"]))
            for p in (0, 1):
                d = restrict(switch_means(rows[m], p)[c], shared)
                s = mean_ci(d, rng)
                out.append(dict(kind="level", model=m, other="", cond=c, phrasing=p,
                                est=s["est"], lo=s["lo"], hi=s["hi"], n=s["n_items"],
                                own_est="", own_n=""))
    for a, b in itertools.combinations(logs, 2):
        for c in CONDS:
            d = paired_diff_ci(restrict(per_model[a][c], shared), restrict(per_model[b][c], shared), rng)
            out.append(dict(kind="paired_diff", model=a, other=b, cond=c, phrasing="all",
                            est=d["est"], lo=d["lo"], hi=d["hi"], n=d["n_items"], own_est="", own_n=""))
    Path("results").mkdir(exist_ok=True)
    with open("results/compare_shared.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        if r["kind"] == "paired_diff":
            print(f"{r['model']} - {r['other']} [{r['cond']}]: {r['est']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}] pp (n={r['n']})")
        elif r["phrasing"] == "all":
            print(f"{r['model']} [{r['cond']}] shared: {r['est']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}] (own known: {r['own_est']:.1f}, n={r['own_n']})")
    figures(out)


def figures(out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(r["model"] for r in out if r["kind"] == "level"))
    lab = [m.replace("ollama/", "") for m in models]
    x = np.arange(len(models))

    def get(m, c, p="all"):
        return next(r for r in out if r["kind"] == "level" and r["model"] == m and r["cond"] == c and r["phrasing"] == p)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (c, name) in enumerate(zip(CONDS, ("mild", "strong"))):
        rs = [get(m, c) for m in models]
        ax.errorbar(x + (k - 0.5) * 0.15, [r["est"] for r in rs],
                    yerr=[[r["est"] - r["lo"] for r in rs], [r["hi"] - r["est"] for r in rs]],
                    fmt="o", capsize=3, label=name)
    ax.set_xticks(x, lab, rotation=15)
    ax.set_ylabel("P(switch to suggested wrong answer), %")
    ax.set_ylim(0, 102)
    ax.set_title(f"Held-out, items known by all models (n={rs[0]['n']})\n95% bootstrap CI over items")
    ax.legend(title="pushback")
    fig.tight_layout()
    fig.savefig("results/fig_primary.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, c, name in zip(axes, CONDS, ("mild", "strong")):
        for p, mk in ((0, "o"), (1, "s")):
            rs = [get(m, c, p) for m in models]
            ax.errorbar(x + (p - 0.5) * 0.15, [r["est"] for r in rs],
                        yerr=[[r["est"] - r["lo"] for r in rs], [r["hi"] - r["est"] for r in rs]],
                        fmt=mk, capsize=3, label=f"phrasing {p}")
        ax.set_xticks(x, lab, rotation=15)
        ax.set_title(name)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("P(switch to suggested wrong answer), %")
    axes[0].legend()
    fig.suptitle("Same strength, different wording (shared items)")
    fig.tight_layout()
    fig.savefig("results/fig_phrasing.png", dpi=150)


if __name__ == "__main__":
    main()
