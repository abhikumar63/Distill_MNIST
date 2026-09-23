"""
The single source of truth for run results (`all_results`), plus every
stats/reporting helper that reads it. Other modules that need to append
or query results import from here — never keep a second copy of the list.
"""

import csv
import math
import os

import numpy as np

import config
from dataset import SPLITS
from metrics import full_metrics

all_results = []

BASELINE_KEY = ("baseline", 0.0, 0.0)


def add_rows(seed, method, T, alpha, logits_by_split, refs):
    for split, (_, labels) in SPLITS.items():
        r = refs[split]
        m = full_metrics(logits_by_split[split], labels, r["prob"], r["logit"], r["dis"])
        all_results.append(
            {"seed": seed, "method": method, "temperature": T, "alpha": alpha, "split": split, **m}
        )


def save_csv():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"kd_{config.MODE}.csv")
    if not all_results:
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    return path


def last_test_row(seed, method, T, alpha):
    return [
        x for x in all_results
        if x["seed"] == seed and x["method"] == method and x["temperature"] == T
        and x["alpha"] == alpha and x["split"] == "test"
    ][-1]


# ============================================================
# Stats helpers
# ============================================================


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), (float(xs.std(ddof=1)) if len(xs) > 1 else float("nan"))


def ci95(d):
    """Paired mean, sd (ddof=1), and t-based 95% CI."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else float("nan")), float("nan"), float("nan"), float("nan")
    m, sd = float(d.mean()), float(d.std(ddof=1))
    h = config.T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, sd, m - h, m + h


def rows_for(method, T, alpha, split):
    return [
        r for r in all_results
        if r["method"] == method and r["temperature"] == T and r["alpha"] == alpha and r["split"] == split
    ]


def paired_diffs(kx, ky, split, metric):
    """x - y over seeds present in both (same seed => same init, same data order)."""
    x = {r["seed"]: r[metric] for r in rows_for(*kx, split)}
    y = {r["seed"]: r[metric] for r in rows_for(*ky, split)}
    return [x[s] - y[s] for s in sorted(set(x) & set(y))]


def fmt(rows, key, digits=4):
    m, s = mean_std([r[key] for r in rows])
    return f"{m:.{digits}f}±{s:.{digits}f}"


def nll_verdict(lo, hi, delta):
    """Verdict for a paired NLL difference X - Y (lower NLL is better)."""
    if math.isnan(lo):
        return "n < 2, no CI"
    if hi < 0:
        return "X better (CI excludes 0)"
    if lo > 0:
        return "Y better (CI excludes 0)"
    if lo > -delta and hi < delta:
        return f"no detectable difference; CI within ±{delta} (equivalent at this delta)"
    return "inconclusive (CI too wide for this delta; more seeds needed)"


REPORT_KEYS = ["acc", "nll", "ece", "dis_acc", "dis_nll", "agree_prob", "kl_prob"]


def print_table(split):
    print(f"\n{'=' * 100}\nMEAN ± STD (ddof=1) | split = {split}\n{'=' * 100}")
    combos = sorted(
        {(r["method"], r["temperature"], r["alpha"]) for r in all_results if r["split"] == split}
    )
    for method, T, alpha in combos:
        rows = rows_for(method, T, alpha, split)
        print(f"\n{method} | T={T} | alpha={alpha} | n={len(rows)}")
        print(
            "  "
            + " | ".join(
                f"{k}: {fmt(rows, k, 3 if k in ('acc', 'dis_acc', 'agree_prob') else 4)}"
                for k in REPORT_KEYS
            )
        )