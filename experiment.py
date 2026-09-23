"""
The two building blocks every workflow (diag/tune/final) is made of:

  prepare_seed(seed) -> trains both teachers, freezes them, precomputes
                         logits and ensemble references for this seed.
  run_student(ctx, method, T, alpha) -> trains one student against ctx,
                         evaluates it, appends rows to results_store.
"""

import numpy as np

import config
from dataset import SPLITS, set_seed, train_images, train_labels
from models import TeacherCNN1, TeacherCNN2, StudentCNN
from train import train_teacher, train_student, crossfit_logits
from metrics import predict_logits, np_softmax
from results_store import add_rows, last_test_row


def prepare_seed(seed):
    """Train teachers for this seed, precompute all logits, build references."""
    print(f"\n{'#' * 80}\nSEED = {seed}\n{'#' * 80}")

    set_seed(seed)
    teacher1 = TeacherCNN1()
    train_teacher(teacher1, seed, "Teacher 1", train_images, train_labels)

    set_seed(seed + 10000)
    teacher2 = TeacherCNN2()
    train_teacher(teacher2, seed + 10000, "Teacher 2", train_images, train_labels)

    train_z = (predict_logits(teacher1, train_images), predict_logits(teacher2, train_images))

    refs, teacher_logits = {}, {}
    for split, (imgs, _) in SPLITS.items():
        z1, z2 = predict_logits(teacher1, imgs), predict_logits(teacher2, imgs)
        teacher_logits[split] = (z1, z2)
        refs[split] = {
            "prob": (np_softmax(z1) + np_softmax(z2)) / 2.0,
            "logit": np_softmax((z1 + z2) / 2.0),
            "dis": z1.argmax(axis=1) != z2.argmax(axis=1),
        }

    add_rows(seed, "ref_teacher_1", 0.0, 0.0, {s: teacher_logits[s][0] for s in SPLITS}, refs)
    add_rows(seed, "ref_teacher_2", 0.0, 0.0, {s: teacher_logits[s][1] for s in SPLITS}, refs)
    add_rows(seed, "ref_ens_prob", 0.0, 0.0, {s: np.log(refs[s]["prob"] + 1e-30) for s in SPLITS}, refs)
    add_rows(
        seed, "ref_ens_logit", 0.0, 0.0,
        {s: (teacher_logits[s][0] + teacher_logits[s][1]) / 2 for s in SPLITS}, refs,
    )

    for name in ("ref_teacher_1", "ref_teacher_2", "ref_ens_prob", "ref_ens_logit"):
        r = last_test_row(seed, name, 0.0, 0.0)
        print(f"{name}: acc={r['acc']:.2f}% nll={r['nll']:.4f} ece={r['ece']:.4f}")
    print(f"Teacher disagreement rate (test): {100 * refs['test']['dis'].mean():.2f}%")

    oof = None
    kd_z = train_z
    if config.CROSS_FIT_FOLDS > 0:
        oof = crossfit_logits(seed)
        kd_z = oof

    del teacher1, teacher2

    return {
        "seed": seed,
        "train_z": train_z,
        "kd_z": kd_z,
        "oof": oof,
        "refs": refs,
        "teacher_logits": teacher_logits,
    }


def run_student(ctx, method, T, alpha, val_log=False):
    """One student. Init + data order depend on the seed only, so every
    method / T / alpha under a seed is a paired comparison."""
    seed = ctx["seed"]
    student_seed = seed + 50000

    set_seed(student_seed)
    student = StudentCNN()

    if method == "baseline":
        train_student(student, "baseline", 1.0, 0.0, student_seed, ctx["kd_z"], val_log)
        store_T, store_alpha = 0.0, 0.0
    else:
        train_student(student, method, T, alpha, student_seed, ctx["kd_z"], val_log)
        store_T, store_alpha = T, alpha

    logits = {s: predict_logits(student, SPLITS[s][0]) for s in SPLITS}
    add_rows(seed, method, store_T, store_alpha, logits, ctx["refs"])

    r = last_test_row(seed, method, store_T, store_alpha)
    print(
        f"RESULT | {method} | T={store_T} alpha={store_alpha} | test acc={r['acc']:.2f}% "
        f"nll={r['nll']:.4f} ece={r['ece']:.4f} | dis_acc={r['dis_acc']:.2f}% "
        f"| agree_prob={r['agree_prob']:.2f}%"
    )