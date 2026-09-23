"""
Training loops for teachers and students, cross-fitting, and a rough
progress/ETA tracker.

Progress state (_t0, _done_units, _planned_units) lives here as module
globals. Callers outside this module set the plan by assigning
`train._planned_units = <value>` before starting a run (see main.py) —
tick()/eta_line() read/update the same module-level names internally.
"""

import time

import numpy as np
import mlx.nn as nn
import mlx.optimizers as optim

import config
from dataset import epoch_batches, set_seed, train_images, train_labels, val_images, val_labels
from models import TeacherCNN1, TeacherCNN2
from losses import cross_entropy, compute_accuracy, kd_kl_loss, make_log_target
from metrics import predict_logits, acc_nll

import mlx.core as mx


# ============================================================
# Progress / ETA (rough: teacher epoch weighted 3x a student epoch)
# ============================================================

_t0 = time.time()
_done_units = 0.0
_planned_units = 1.0


def tick(w):
    global _done_units
    _done_units += w


def eta_line():
    el = time.time() - _t0
    frac = min(_done_units / max(_planned_units, 1e-9), 1.0)
    eta = el / frac - el if frac > 0 else float("nan")
    print(f"  [progress {100 * frac:.0f}% | elapsed {el / 60:.1f}m | ETA ~{eta / 60:.1f}m]")


def teacher_units():
    per_seed = 2 + (2 * config.CROSS_FIT_FOLDS if config.CROSS_FIT_FOLDS > 0 else 0)
    return per_seed * config.TEACHER_EPOCHS * config.TEACHER_W


# ============================================================
# Training
# ============================================================


def ce_loss_fn(model, x, y):
    logits = model(x)
    return cross_entropy(logits, y), logits


def train_teacher(model, seed, name, images, labels, verbose=True):
    if verbose:
        print(f"\n{'=' * 70}\nTRAINING {name}\n{'=' * 70}")
    opt = optim.Adam(learning_rate=config.LR_TEACHER)
    loss_and_grad = nn.value_and_grad(model, ce_loss_fn)

    for epoch in range(config.TEACHER_EPOCHS):
        tot_loss = tot_acc = n = 0.0
        for b in epoch_batches(len(images), seed, epoch):
            x, y = mx.array(images[b]), mx.array(labels[b])
            (loss, logits), grads = loss_and_grad(model, x, y)
            opt.update(model, grads)
            acc = compute_accuracy(logits, y)
            mx.eval(model.parameters(), opt.state, loss, acc)
            tot_loss += float(loss) * len(b)
            tot_acc += float(acc) * len(b)
            n += len(b)
        tick(config.TEACHER_W)
        if verbose or epoch == config.TEACHER_EPOCHS - 1:
            print(
                f"  {name} epoch {epoch + 1}/{config.TEACHER_EPOCHS} "
                f"| Loss: {tot_loss / n:.4f} | Acc: {100 * tot_acc / n:.2f}%"
            )

    # Frozen + eval. Logits are precomputed in NumPy so no gradient can
    # reach the teachers anyway; this is hygiene.
    model.freeze()
    model.eval()
    eta_line()


def train_student(student, method, T, alpha, seed, teacher_train_logits, val_log=False):
    """
    method: "baseline" (CE only) or one of config.KD_METHODS.
    Init and data order depend only on `seed`, never on method/T/alpha.
    """
    tag = "baseline" if method == "baseline" else f"{method}, T={T}, alpha={alpha}"
    print(f"\nTraining {tag}")
    opt = optim.Adam(learning_rate=config.LR_STUDENT)
    z1_all, z2_all = teacher_train_logits

    def kd_loss_fn(model, x, y, log_target):
        logits = model(x)
        ce = cross_entropy(logits, y)
        kl = kd_kl_loss(log_target, logits, T)
        total = alpha * ce + (1.0 - alpha) * (T**2) * kl
        return total, ce, kl, logits

    kd_and_grad = nn.value_and_grad(student, kd_loss_fn)
    ce_and_grad = nn.value_and_grad(student, ce_loss_fn)

    for epoch in range(config.STUDENT_EPOCHS):
        tot = tot_ce = tot_kl = tot_acc = n = 0.0
        for b in epoch_batches(len(train_images), seed, epoch):
            x, y = mx.array(train_images[b]), mx.array(train_labels[b])

            if method == "baseline":
                (loss, logits), grads = ce_and_grad(student, x, y)
                ce, kl = loss, mx.array(0.0)
            else:
                lt = make_log_target(method, mx.array(z1_all[b]), mx.array(z2_all[b]), T)
                (loss, ce, kl, logits), grads = kd_and_grad(student, x, y, lt)

            opt.update(student, grads)
            acc = compute_accuracy(logits, y)
            mx.eval(student.parameters(), opt.state, loss, ce, kl, acc)

            k = len(b)
            tot += float(loss) * k
            tot_ce += float(ce) * k
            tot_kl += float(kl) * k
            tot_acc += float(acc) * k
            n += k
        tick(config.STUDENT_W)

        line = (
            f"  Epoch {epoch + 1}/{config.STUDENT_EPOCHS} | Total: {tot / n:.4f} "
            f"| CE: {tot_ce / n:.4f} | KL: {tot_kl / n:.4f} | Acc: {100 * tot_acc / n:.2f}%"
        )
        if val_log:
            va, vn = acc_nll(predict_logits(student, val_images), val_labels)
            line += f" | val acc {va:.2f}% nll {vn:.4f}"
        print(line)

    eta_line()


# ============================================================
# Cross-fitting: out-of-fold teacher logits for the train set
# ============================================================


def crossfit_logits(seed):
    K = config.CROSS_FIT_FOLDS
    n = len(train_images)
    perm = np.random.default_rng(seed + 777).permutation(n)
    folds = np.array_split(perm, K)

    # folds must partition the train set
    assert sum(len(f) for f in folds) == n and len(np.unique(np.concatenate(folds))) == n

    oof1 = np.zeros((n, 10), dtype=np.float32)
    oof2 = np.zeros((n, 10), dtype=np.float32)

    print(f"\nCross-fitting teachers: {K} folds")
    for k, hold in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(K) if j != k])

        set_seed(seed + 2000 + k)
        t1 = TeacherCNN1()
        train_teacher(t1, seed + 2000 + k, f"T1 fold {k + 1}/{K}", train_images[tr], train_labels[tr], verbose=False)
        oof1[hold] = predict_logits(t1, train_images[hold])

        set_seed(seed + 3000 + k)
        t2 = TeacherCNN2()
        train_teacher(t2, seed + 3000 + k, f"T2 fold {k + 1}/{K}", train_images[tr], train_labels[tr], verbose=False)
        oof2[hold] = predict_logits(t2, train_images[hold])

    return oof1, oof2