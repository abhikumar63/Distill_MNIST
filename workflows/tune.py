"""MODE = "tune": grid-search alpha x T on val, gate KD against baseline,
write results/best_config.json with the selected configs and a seed-count
recommendation for MODE = "final"."""

import json
import math
import os

import config
from experiment import prepare_seed, run_student
from results_store import rows_for, mean_std, paired_diffs, ci95, BASELINE_KEY, save_csv


def run_tune():
    for seed in config.TUNE_SEEDS:
        ctx = prepare_seed(seed)
        run_student(ctx, "baseline", 0.0, 0.0)
        for method in config.TUNE_METHODS:
            for alpha in config.ALPHAS:
                for T in config.TEMPERATURES:
                    run_student(ctx, method, T, alpha)
        print(f"\nSaved {save_csv()}")
    report_tune()


def report_tune():
    base_rows = rows_for(*BASELINE_KEY, "val")
    base_nll = mean_std([r["nll"] for r in base_rows])[0]
    base_acc = mean_std([r["acc"] for r in base_rows])[0]

    best = {}
    for method in config.TUNE_METHODS:
        table = {}
        for alpha in config.ALPHAS:
            for T in config.TEMPERATURES:
                rows = rows_for(method, T, alpha, "val")
                table[(alpha, T)] = mean_std([r["nll"] for r in rows])[0]
        (b_alpha, b_T), b_nll = min(table.items(), key=lambda kv: kv[1])
        best[method] = {"alpha": b_alpha, "T": b_T, "val_nll": b_nll}

        print(f"\n{'=' * 100}\n{method}: mean VAL NLL over {len(config.TUNE_SEEDS)} seeds "
              f"(baseline {base_nll:.4f}); * = best\n{'=' * 100}")
        print("alpha \\ T " + "".join(f"{T:>10}" for T in config.TEMPERATURES))
        for alpha in config.ALPHAS:
            cells = ""
            for T in config.TEMPERATURES:
                mark = "*" if (alpha, T) == (b_alpha, b_T) else " "
                cells += f"{table[(alpha, T)]:>9.4f}{mark}"
            print(f"{alpha:<9}  {cells}")

    # Gate: does KD beat the CE-only baseline at all?
    print(f"\n{'=' * 100}\nGATE: does KD beat the CE-only baseline on VAL?\n{'=' * 100}")
    gate_ok = True
    for method in config.TUNE_METHODS:
        k = (method, best[method]["T"], best[method]["alpha"])
        d = paired_diffs(k, BASELINE_KEY, "val", "nll")
        m, sd, lo, hi = ci95(d)
        wins = sum(x < 0 for x in d)
        ok = m < 0
        gate_ok &= ok
        print(
            f"{method} (alpha={k[2]}, T={k[1]}): dNLL vs baseline {m:+.4f} (sd {sd:.4f}) "
            f"| KD lower NLL in {wins}/{len(d)} seeds | {'PASS' if ok else 'FAIL'}"
        )
    print(f"baseline val: acc {base_acc:.2f}% nll {base_nll:.4f}")
    if not gate_ok:
        print(
            "\n!! KD does not beat the baseline on val. A-vs-B is comparing two ways of not "
            "helping.\n   Try: CROSS_FIT_FOLDS = 4, more STUDENT_EPOCHS, or a wider ALPHAS/TEMPERATURES "
            "grid before running final."
        )

    # Paired A - B at every common grid point (val)
    if "kd_A" in config.TUNE_METHODS and "kd_B" in config.TUNE_METHODS:
        print(f"\n{'=' * 100}\nPAIRED A - B on VAL at each grid point (dNLL, negative = A better)\n{'=' * 100}")
        for alpha in config.ALPHAS:
            for T in config.TEMPERATURES:
                d = paired_diffs(("kd_A", T, alpha), ("kd_B", T, alpha), "val", "nll")
                m, sd, _, _ = ci95(d)
                print(f"alpha={alpha:<4} T={T:<4} dNLL {m:+.4f} ± {sd:.4f}")

        kA = ("kd_A", best["kd_A"]["T"], best["kd_A"]["alpha"])
        kB = ("kd_B", best["kd_B"]["T"], best["kd_B"]["alpha"])
        d = paired_diffs(kA, kB, "val", "nll")
        m, sd, _, _ = ci95(d)
        print(f"\nA@best vs B@best on VAL: dNLL {m:+.4f} ± {sd:.4f} (n={len(d)})")
        rec = None
        if not math.isnan(sd) and sd > 0:
            rec = int(math.ceil(8.0 * (sd / config.MIN_EFFECT_NLL) ** 2))
            print(
                f"Seeds needed for ~80% power to detect |dNLL| = {config.MIN_EFFECT_NLL}: ~{rec} "
                f"(= 8*(sd/delta)^2). sd comes from only {len(d)} pilot seeds, so treat as rough."
            )
        best["recommended_seeds"] = rec
        best["pilot_sd_dNLL_val"] = None if math.isnan(sd) else sd

    best["baseline_val_nll"] = base_nll
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.BEST_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved {config.BEST_PATH}")
    print("Next: set FINAL_SEEDS to at least the recommended size and run MODE = \"final\".")