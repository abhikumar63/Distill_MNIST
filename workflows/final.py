"""MODE = "final": run the val-selected configs across FINAL_SEEDS and
report the paired primary metric (test NLL, A - B) plus secondary metrics."""

import config
from experiment import prepare_seed, run_student
from results_store import (
    print_table, paired_diffs, ci95, nll_verdict, rows_for, mean_std, BASELINE_KEY, save_csv,
)


def final_jobs(best):
    aA, TA = best["kd_A"]["alpha"], best["kd_A"]["T"]
    aB, TB = best["kd_B"]["alpha"], best["kd_B"]["T"]

    jobs = {("kd_A", TA, aA), ("kd_B", TB, aB)}
    if config.INCLUDE_SINGLE_TEACHERS:
        jobs |= {("kd_T1", TA, aA), ("kd_T2", TA, aA)}
    if config.FINAL_T_SWEEP:
        for T in config.TEMPERATURES:
            jobs |= {("kd_A", T, aA), ("kd_B", T, aA)}
    return sorted(jobs, key=lambda j: (j[0], j[1], j[2]))


def run_final(best, jobs):
    for seed in config.FINAL_SEEDS:
        ctx = prepare_seed(seed)
        run_student(ctx, "baseline", 0.0, 0.0)
        for method, T, alpha in jobs:
            run_student(ctx, method, T, alpha)
        print(f"\nSaved {save_csv()}")
    report_final(best)


def report_final(best):
    aA, TA = best["kd_A"]["alpha"], best["kd_A"]["T"]
    aB, TB = best["kd_B"]["alpha"], best["kd_B"]["T"]
    kA, kB = ("kd_A", TA, aA), ("kd_B", TB, aB)

    print_table("test")

    # ---- PRIMARY ----
    d = paired_diffs(kA, kB, "test", "nll")
    m, sd, lo, hi = ci95(d)
    print(f"\n{'=' * 100}\nPRIMARY: paired test NLL, A@best - B@best (negative = A better)\n{'=' * 100}")
    print(f"A = alpha {aA}, T {TA} | B = alpha {aB}, T {TB} (each at its own val-selected config)")
    print(f"dNLL = {m:+.4f} | sd {sd:.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] | n = {len(d)}")
    print(f"A lower NLL in {sum(x < 0 for x in d)}/{len(d)} seeds")
    print("Verdict (X=A, Y=B):", nll_verdict(lo, hi, config.MIN_EFFECT_NLL))
    rec = best.get("recommended_seeds")
    if rec and len(d) < rec:
        print(f"!! n={len(d)} is below the ~{rec} seeds the pilot suggested; an 'inconclusive' verdict is expected.")

    # ---- KD vs baseline ----
    print(f"\n{'=' * 100}\nKD vs CE-only baseline (paired, test)\n{'=' * 100}")
    for name, k in (("A", kA), ("B", kB)):
        for metric in ("nll", "acc"):
            dd = paired_diffs(k, BASELINE_KEY, "test", metric)
            mm, ss, l, h = ci95(dd)
            print(f"{name} - baseline {metric}: {mm:+.4f} | 95% CI [{l:+.4f}, {h:+.4f}] | n={len(dd)}")

    # ---- Secondary (exploratory) ----
    print(f"\n{'=' * 100}\nSECONDARY (exploratory): A@best - B@best\n{'=' * 100}")
    for metric in ("acc", "ece", "dis_acc", "dis_nll", "kl_prob", "kl_logit"):
        dd = paired_diffs(kA, kB, "test", metric)
        mm, ss, l, h = ci95(dd)
        print(f"{metric:<9} {mm:+.4f} | 95% CI [{l:+.4f}, {h:+.4f}]")
    dis = mean_std([r["dis_frac"] for r in rows_for(*BASELINE_KEY, "test")])[0]
    print(f"Teacher disagreement rate (test, mean over seeds): {dis:.2f}%")

    # ---- T-curve at fixed alpha ----
    if config.FINAL_T_SWEEP:
        print(f"\n{'=' * 100}\nT-CURVE at alpha={aA}: paired A - B, test dNLL\n{'=' * 100}")
        for T in config.TEMPERATURES:
            dd = paired_diffs(("kd_A", T, aA), ("kd_B", T, aA), "test", "nll")
            mm, ss, l, h = ci95(dd)
            print(f"T={T:<4} dNLL {mm:+.4f} | 95% CI [{l:+.4f}, {h:+.4f}] | n={len(dd)}")