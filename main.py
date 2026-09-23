"""
Entry point. Set MODE in config.py, then:

    python main.py

RUN IN THIS ORDER:
  1. MODE = "diag"   ~minutes. Checks (a) whether teacher targets on train are
                     in-sample, (b) whether the student is still improving at
                     STUDENT_EPOCHS, (c) how different A and B targets are.
  2. MODE = "tune"   Pilot on TUNE_SEEDS (different from FINAL_SEEDS). Grid over
                     ALPHAS x TEMPERATURES on VAL. Writes results/best_config.json,
                     prints a KD-vs-baseline gate and a seed-count recommendation.
  3. MODE = "final"  FINAL_SEEDS at the selected configs. Reports the PRIMARY
                     metric (paired test NLL, A - B, 95% CI) and secondary metrics.

Test data is touched for reporting only; every selection uses VAL.
"""

import json
import os

import config
import sanity
import train  # for _planned_units / teacher_units()
import dataset  # noqa: F401  (import triggers the Fashion-MNIST load exactly once)

from workflows import diag, tune, final


if __name__ == "__main__":
    sanity.sanity_checks()

    if config.MODE == "diag":
        train._planned_units = train.teacher_units() + 1 * config.STUDENT_EPOCHS * config.STUDENT_W
        diag.run_diag()

    elif config.MODE == "tune":
        assert "kd_A" in config.TUNE_METHODS and "kd_B" in config.TUNE_METHODS, "tune both A and B"
        n_students = 1 + len(config.TUNE_METHODS) * len(config.ALPHAS) * len(config.TEMPERATURES)
        train._planned_units = len(config.TUNE_SEEDS) * (
            train.teacher_units() + n_students * config.STUDENT_EPOCHS * config.STUDENT_W
        )
        print(f"\nTune plan: {len(config.TUNE_SEEDS)} seeds x {n_students} students")
        tune.run_tune()

    elif config.MODE == "final":
        if not os.path.exists(config.BEST_PATH):
            raise SystemExit(f"{config.BEST_PATH} not found. Run MODE = 'tune' first.")
        with open(config.BEST_PATH) as f:
            best = json.load(f)
        jobs = final.final_jobs(best)
        n_students = 1 + len(jobs)
        train._planned_units = len(config.FINAL_SEEDS) * (
            train.teacher_units() + n_students * config.STUDENT_EPOCHS * config.STUDENT_W
        )
        print(f"\nFinal plan: {len(config.FINAL_SEEDS)} seeds x {n_students} students; jobs: {jobs}")
        final.run_final(best, jobs)

    else:
        raise SystemExit(f"Unknown MODE {config.MODE!r}")