"""
NumPy-side evaluation: softmax/NLL/ECE/agreement/KL, plus the MLX
inference helper that turns a trained model into logits.
"""

import numpy as np
import mlx.core as mx


def np_softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def acc_nll(logits, labels):
    p = np_softmax(logits)
    acc = 100.0 * float(np.mean(p.argmax(axis=1) == labels))
    nll = float(-np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12)))
    return acc, nll


def compute_ece(probs, labels, n_bins=15):
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def kl_to_ref(ref_probs, probs):
    """Mean KL(ref || model) at T=1 over the split."""
    eps = 1e-12
    return float(
        np.mean(np.sum(ref_probs * (np.log(ref_probs + eps) - np.log(probs + eps)), axis=1))
    )


def full_metrics(logits, labels, ref_prob, ref_logit, dis_mask):
    """
    ref_prob : prob-mean teacher ensemble (B-style), T=1
    ref_logit: logit-mean teacher ensemble (A-style), T=1
    dis_mask : examples where the two teachers' argmax differ
    """
    p = np_softmax(logits)
    pred = p.argmax(axis=1)
    idx = np.arange(len(labels))

    m = {
        "acc": 100.0 * float(np.mean(pred == labels)),
        "nll": float(-np.mean(np.log(p[idx, labels] + 1e-12))),
        "ece": compute_ece(p, labels),
        "agree_prob": 100.0 * float(np.mean(pred == ref_prob.argmax(axis=1))),
        "agree_logit": 100.0 * float(np.mean(pred == ref_logit.argmax(axis=1))),
        "kl_prob": kl_to_ref(ref_prob, p),
        "kl_logit": kl_to_ref(ref_logit, p),
        "dis_frac": 100.0 * float(dis_mask.mean()),
    }

    if dis_mask.any():
        d = dis_mask
        m["dis_acc"] = 100.0 * float(np.mean(pred[d] == labels[d]))
        m["dis_nll"] = float(-np.mean(np.log(p[d][np.arange(d.sum()), labels[d]] + 1e-12)))
    else:
        m["dis_acc"] = float("nan")
        m["dis_nll"] = float("nan")

    return m


def predict_logits(model, images, chunk=1000):
    out = []
    for s in range(0, len(images), chunk):
        logits = model(mx.array(images[s : s + chunk]))
        mx.eval(logits)
        out.append(np.array(logits))
    return np.concatenate(out, axis=0)