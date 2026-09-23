# KD Ensemble Experiment (Fashion-MNIST, MLX)

Does it matter *how* you combine two teachers when distilling into a student?

- **Method A** (logit average): `softmax(mean(z1, z2) / T)`
- **Method B** (prob average): `mean(softmax(z1/T), softmax(z2/T))`

Loss: `L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(teacher_T || student_T)`

## Setup

```bash
pip install mlx torchvision numpy
```

## Run

Set `MODE` in `config.py`, then:

```bash
python main.py
```

Run in order:

1. **`MODE = "diag"`** — one seed, fast. Checks whether teacher targets are
   in-sample-biased, whether `STUDENT_EPOCHS` is enough, and how different
   A and B actually are.
2. **`MODE = "tune"`** — grid search `ALPHAS x TEMPERATURES` on a val split,
   for both methods independently. Gates KD against a CE-only baseline.
   Writes `results/best_config.json` with the selected configs and a
   recommended seed count for step 3.
3. **`MODE = "final"`** — runs the selected configs across `FINAL_SEEDS`.
   Reports the primary metric (paired test NLL, A − B, 95% CI) plus
   secondary metrics (accuracy, ECE, disagreement-subset, T-curve).

## Layout

```
main.py            entry point / mode dispatch
config.py           all knobs
dataset.py          Fashion-MNIST load, train/val/test split, batching
models.py            teacher/student CNNs
losses.py            CE, KD targets (A/B), KL loss
metrics.py            NLL, ECE, agreement, predict_logits
sanity.py             unit tests for the loss math (no data)
train.py              training loops, cross-fitting, progress/ETA
experiment.py          per-seed setup + per-student run (shared by all modes)
results_store.py       results table, CSV, paired stats, CI
workflows/diag.py       MODE = "diag"
workflows/tune.py       MODE = "tune"
workflows/final.py      MODE = "final"
```

## Output

- `results/kd_{diag,tune,final}.csv` — every run's metrics
- `results/best_config.json` — tune's selected (alpha, T) per method

## Notes

- Every student under one seed shares init + batch order (`seed + 50000`),
  so A vs B is a paired comparison, not an independent one.
- `CROSS_FIT_FOLDS = 0` by default (in-sample teacher targets). `diag`
  mode tells you whether to turn it on.# Distill_MNIST
