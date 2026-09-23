"""
All tunable knobs for the KD ensemble experiment. No functions here —
every other module imports this one, so keep it dependency-free.
"""

import os

MODE = "final"  # "diag" -> "tune" -> "final"

BATCH_SIZE = 128
TEACHER_EPOCHS = 5
STUDENT_EPOCHS = 15  # students were still improving at 5; baseline gets the same
LR_TEACHER = 1e-3
LR_STUDENT = 1e-3

# Held out from the 60k train set. Used ONLY for selection (T, alpha).
VAL_SIZE = 5000

# 0 = KD targets on train come from the teachers' own training set (in-sample).
# K > 0 = out-of-fold teacher logits (each train example is scored by teachers
# that never saw it). Costs 2*K extra teacher trainings per seed.
# Decide after MODE = "diag" (see the train/val NLL ratio it prints).
CROSS_FIT_FOLDS = 0

ALPHAS = [0.1, 0.5, 0.9]
TEMPERATURES = [1.0, 2.0, 4.0, 8.0]

DIAG_SEED = 42
TUNE_SEEDS = [1001, 1002, 1003]  # disjoint from FINAL_SEEDS on purpose
TUNE_METHODS = ["kd_A", "kd_B"]  # both must be tuned; each gets its own best config
FINAL_SEEDS = [
    42, 123, 456, 789, 2026, 3, 5, 8, 13, 21,
    34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,
    7, 11, 17, 19, 23, 29, 31, 37, 41, 43,
    47, 53, 59, 61, 67,
]  # extend to the size tune recommends

INCLUDE_SINGLE_TEACHERS = True  # kd_T1 / kd_T2 at kd_A's selected (alpha, T)
FINAL_T_SWEEP = True  # kd_A and kd_B across TEMPERATURES at kd_A's alpha (paired T-curve)

# Smallest test-NLL difference you would care about. Used for the seed-count
# recommendation and the "equivalent within +-delta" verdict. Your call.
MIN_EFFECT_NLL = 0.005

KD_METHODS = ["kd_T1", "kd_T2", "kd_A", "kd_B"]
RESULTS_DIR = "results"
BEST_PATH = os.path.join(RESULTS_DIR, "best_config.json")

TEACHER_W = 3.0  # progress-bar weighting: one teacher epoch counts as 3 student epochs
STUDENT_W = 1.0

T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}