"""MODE = "diag": one seed, fast sanity run before spending real compute."""

import numpy as np

import config
from dataset import train_labels, val_labels
from experiment import prepare_seed, run_student
from metrics import acc_nll, kl_to_ref


def teacher_diagnostics(ctx):
    print(f"\n{'=' * 100}\nTEACHER IN-SAMPLE DIAGNOSTIC\n{'=' * 100}")
    print("train = what KD targets are computed on (in-sample unless cross-fit); val = held out.")

    ratios = []
    for i, name in enumerate(("T1", "T2")):
        tr_a, tr_n = acc_nll(ctx["train_z"][i], train_labels)
        va_a, va_n = acc_nll(ctx["teacher_logits"]["val"][i], val_labels)
        ratio = tr_n / va_n
        ratios.append(ratio)
        line = (
            f"{name}: train acc {tr_a:.2f}% nll {tr_n:.4f} | val acc {va_a:.2f}% nll {va_n:.4f} "
            f"| train/val NLL ratio {ratio:.2f}"
        )
        if ctx["oof"] is not None:
            oa, on = acc_nll(ctx["oof"][i], train_labels)
            line += f" | out-of-fold acc {oa:.2f}% nll {on:.4f}"
        print(line)

    if config.CROSS_FIT_FOLDS == 0:
        if min(ratios) < 0.6:
            print(
                "\n-> Train NLL is well below val NLL (rule of thumb: ratio < 0.6). KD targets are "
                "overconfident on train.\n   Set CROSS_FIT_FOLDS = 4 and re-run diag/tune/final."
            )
        else:
            print("\n-> In-sample gap is small. CROSS_FIT_FOLDS = 0 is defensible.")
    else:
        print(
            "\n-> Cross-fit on. Out-of-fold NLL should be close to val NLL; that is what the "
            "students now distill from."
        )

    # How different are the A and B targets at T=1 (top-1 and distribution)?
    refp, refl = ctx["refs"]["test"]["prob"], ctx["refs"]["test"]["logit"]
    top1 = 100.0 * float(np.mean(refp.argmax(axis=1) == refl.argmax(axis=1)))
    print(
        f"\nA vs B reference ensembles on test: top-1 agreement {top1:.2f}% | "
        f"KL(prob-mean || logit-mean) {kl_to_ref(refp, refl):.4f} nats"
    )
    print("A small gap here bounds how large an A-vs-B student difference can be.")


def run_diag():
    ctx = prepare_seed(config.DIAG_SEED)
    teacher_diagnostics(ctx)
    print(f"\nBaseline student with per-epoch val log ({config.STUDENT_EPOCHS} epochs):")
    run_student(ctx, "baseline", 0.0, 0.0, val_log=True)
    print(
        "\n-> If val NLL is still falling at the last epoch, raise STUDENT_EPOCHS. "
        "If it bottoms out and rises, lower it."
        "\n-> Then: set CROSS_FIT_FOLDS per the diagnostic above and run MODE = \"tune\"."
    )